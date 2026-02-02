"""
File Processing Orchestrator
=============================
Coordinates document processing workflow with Recursive Context Stitching.
Implements contextual bridging between pages for coherent multi-page documents.
"""

import asyncio
import uuid
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import chainlit as cl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from config.settings import get_config
from processing.document_extractor import (
    extract_pdf_pages,
    extract_pdf_pages_with_layout,
    extract_pdf_images,
    extract_docx,
    extract_txt,
    PageLayout
)
from processing.vision import analyze_image
from rag.embeddings import embed_documents
from storage.vector_store import add_chunks_to_collection
from rag.semantic_chunker import get_hybrid_chunker, smart_split_text

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Hybrid chunker (semantic + character fallback)
# Uses semantic chunking for long texts, character-based for short texts
hybrid_chunker = get_hybrid_chunker()

# Legacy character-based splitter (for fallback or comparison)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunking.chunk_size,
    chunk_overlap=config.chunking.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Thread pool (original logic)
executor = ThreadPoolExecutor(
    max_workers=config.performance.file_processing_workers
)

# Reasoning client for context summarization
reasoning_client = AsyncOpenAI(
    base_url=config.models.reasoning_base_url,
    api_key="EMPTY"
)

# =============================================================================
# RECURSIVE CONTEXT STITCHING
# =============================================================================

@dataclass
class PageContext:
    """Context information extracted from a page for bridging."""
    page_number: int
    key_entities: List[str]  # Named entities: people, places, organizations
    themes: List[str]  # Main topics/themes
    summary: str  # Brief page summary
    unresolved_references: List[str]  # References that need context from next pages
    continuing_topics: List[str]  # Topics that likely continue on next page

@dataclass
class DocumentContext:
    """Accumulated context across all processed pages."""
    filename: str
    page_contexts: Dict[int, PageContext] = field(default_factory=dict)
    global_entities: Dict[str, List[int]] = field(default_factory=dict)  # entity -> pages
    document_themes: List[str] = field(default_factory=list)

# Context extraction prompts
CONTEXT_EXTRACTION_PROMPT = """Analyze this page content and extract contextual information.

PAGE {page_num} CONTENT:
{content}

PREVIOUS PAGE CONTEXT (if any):
{prev_context}

Extract:
1. KEY_ENTITIES: List important names, places, organizations, dates, numerical values
2. THEMES: Main topics discussed on this page (3-5 themes max)
3. SUMMARY: 2-3 sentence summary of this page
4. UNRESOLVED: References that seem incomplete or need more context
5. CONTINUING: Topics that appear to continue on the next page

Format your response as:
KEY_ENTITIES: entity1, entity2, entity3
THEMES: theme1, theme2, theme3
SUMMARY: Your summary here.
UNRESOLVED: item1, item2
CONTINUING: topic1, topic2"""

CONTEXT_BRIDGE_TEMPLATE = """[CONTEXT FROM PREVIOUS PAGES]
Document themes: {themes}
Key entities: {entities}
Previous page summary: {prev_summary}
Continuing topics: {continuing}
[END PREVIOUS CONTEXT]

"""

# Batch processing constants
CONTEXT_BATCH_SIZE = 5   # Pages per LLM call (reduces 78 calls to ~16)
MIN_PAGE_LENGTH = 100    # Skip context extraction for very short pages
IMAGE_CONCURRENCY = 4    # Concurrent image analyses

BATCH_CONTEXT_EXTRACTION_PROMPT = """Analyze each page below and extract contextual information for each one.

PREVIOUS CONTEXT FROM EARLIER PAGES:
{prev_context}

{pages_content}

For EACH page ({page_numbers}), provide output in this EXACT format:

=== PAGE [number] ===
KEY_ENTITIES: entity1, entity2, entity3
THEMES: theme1, theme2, theme3
SUMMARY: 2-3 sentence summary.
UNRESOLVED: item1, item2
CONTINUING: topic1, topic2

Output one block per page. Keep summaries concise."""


async def extract_batch_page_contexts(
    pages: List[Tuple[int, str]],
    prev_context: Optional[PageContext] = None
) -> List[PageContext]:
    """
    Extract contextual information from multiple pages in a single LLM call.
    Reduces N sequential calls to 1 call per batch.
    """
    if not pages:
        return []

    try:
        prev_context_str = ""
        if prev_context:
            prev_context_str = (
                f"Previous Summary: {prev_context.summary}\n"
                f"Key Entities: {', '.join(prev_context.key_entities[:10])}\n"
                f"Continuing Topics: {', '.join(prev_context.continuing_topics)}"
            )

        # Build multi-page content
        pages_content = ""
        for page_num, content in pages:
            pages_content += f"\n--- PAGE {page_num} ---\n{content[:3000]}\n"

        response = await reasoning_client.chat.completions.create(
            model=config.models.reasoning_model,
            messages=[{
                "role": "user",
                "content": BATCH_CONTEXT_EXTRACTION_PROMPT.format(
                    prev_context=prev_context_str if prev_context_str else "None (first batch)",
                    pages_content=pages_content,
                    page_numbers=", ".join(str(pn) for pn, _ in pages)
                )
            }],
            max_tokens=min(256 * len(pages), 2048),
            temperature=0.2
        )

        result = response.choices[0].message.content or ""

        # Parse response into individual page contexts
        contexts = []
        page_sections = re.split(r'===\s*PAGE\s+(\d+)\s*===', result)

        # page_sections alternates: [preamble, page_num, content, page_num, content, ...]
        for i in range(1, len(page_sections) - 1, 2):
            page_num = int(page_sections[i])
            section = page_sections[i + 1]

            entities = []
            themes = []
            summary = ""
            unresolved = []
            continuing = []

            for line in section.split('\n'):
                line = line.strip()
                if line.startswith('KEY_ENTITIES:'):
                    entities = [e.strip() for e in line[13:].split(',') if e.strip()]
                elif line.startswith('THEMES:'):
                    themes = [t.strip() for t in line[7:].split(',') if t.strip()]
                elif line.startswith('SUMMARY:'):
                    summary = line[8:].strip()
                elif line.startswith('UNRESOLVED:'):
                    unresolved = [u.strip() for u in line[11:].split(',') if u.strip()]
                elif line.startswith('CONTINUING:'):
                    continuing = [c.strip() for c in line[11:].split(',') if c.strip()]

            contexts.append(PageContext(
                page_number=page_num,
                key_entities=entities[:20],
                themes=themes[:5],
                summary=summary[:500],
                unresolved_references=unresolved[:10],
                continuing_topics=continuing[:5]
            ))

        # If parsing failed, return empty contexts for each page
        if not contexts:
            return [PageContext(
                page_number=pn, key_entities=[], themes=[],
                summary="", unresolved_references=[], continuing_topics=[]
            ) for pn, _ in pages]

        return contexts

    except Exception as e:
        print(f"⚠️ Batch context extraction error: {e}")
        return [PageContext(
            page_number=pn, key_entities=[], themes=[],
            summary="", unresolved_references=[], continuing_topics=[]
        ) for pn, _ in pages]


async def extract_page_context(
    page_num: int,
    content: str,
    prev_context: Optional[PageContext] = None
) -> PageContext:
    """
    Extract contextual information from a page for bridging to subsequent pages.
    """
    try:
        prev_context_str = ""
        if prev_context:
            prev_context_str = f"""
Previous Page Summary: {prev_context.summary}
Previous Key Entities: {', '.join(prev_context.key_entities[:10])}
Continuing Topics: {', '.join(prev_context.continuing_topics)}
"""

        response = await reasoning_client.chat.completions.create(
            model=config.models.reasoning_model,
            messages=[{
                "role": "user",
                "content": CONTEXT_EXTRACTION_PROMPT.format(
                    page_num=page_num,
                    content=content[:3000],  # Limit content length
                    prev_context=prev_context_str
                )
            }],
            max_tokens=256,
            temperature=0.2
        )

        result = response.choices[0].message.content or ""

        # Parse response
        entities = []
        themes = []
        summary = ""
        unresolved = []
        continuing = []

        for line in result.split('\n'):
            line = line.strip()
            if line.startswith('KEY_ENTITIES:'):
                entities = [e.strip() for e in line[13:].split(',') if e.strip()]
            elif line.startswith('THEMES:'):
                themes = [t.strip() for t in line[7:].split(',') if t.strip()]
            elif line.startswith('SUMMARY:'):
                summary = line[8:].strip()
            elif line.startswith('UNRESOLVED:'):
                unresolved = [u.strip() for u in line[11:].split(',') if u.strip()]
            elif line.startswith('CONTINUING:'):
                continuing = [c.strip() for c in line[11:].split(',') if c.strip()]

        return PageContext(
            page_number=page_num,
            key_entities=entities[:20],  # Limit entities
            themes=themes[:5],
            summary=summary[:500],
            unresolved_references=unresolved[:10],
            continuing_topics=continuing[:5]
        )

    except Exception as e:
        print(f"⚠️ Context extraction error (p{page_num}): {e}")
        return PageContext(
            page_number=page_num,
            key_entities=[],
            themes=[],
            summary="",
            unresolved_references=[],
            continuing_topics=[]
        )

def build_context_bridge(
    doc_context: DocumentContext,
    current_page: int
) -> str:
    """
    Build a context bridge string from previous pages for the current page.
    """
    if current_page <= 1 or not doc_context.page_contexts:
        return ""

    prev_page = current_page - 1
    prev_context = doc_context.page_contexts.get(prev_page)

    if not prev_context:
        return ""

    # Collect global themes (from all pages)
    all_themes = set()
    for pc in doc_context.page_contexts.values():
        all_themes.update(pc.themes)
    themes_str = ', '.join(list(all_themes)[:10])

    # Get frequently mentioned entities
    frequent_entities = sorted(
        doc_context.global_entities.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:15]
    entities_str = ', '.join([e for e, _ in frequent_entities])

    return CONTEXT_BRIDGE_TEMPLATE.format(
        themes=themes_str if themes_str else "Not yet determined",
        entities=entities_str if entities_str else "None identified",
        prev_summary=prev_context.summary if prev_context.summary else "No summary available",
        continuing=', '.join(prev_context.continuing_topics) if prev_context.continuing_topics else "None"
    )

def update_document_context(
    doc_context: DocumentContext,
    page_context: PageContext
):
    """
    Update the global document context with a new page's context.
    """
    doc_context.page_contexts[page_context.page_number] = page_context

    # Update global entities
    for entity in page_context.key_entities:
        if entity not in doc_context.global_entities:
            doc_context.global_entities[entity] = []
        doc_context.global_entities[entity].append(page_context.page_number)

    # Update document themes
    for theme in page_context.themes:
        if theme not in doc_context.document_themes:
            doc_context.document_themes.append(theme)

# =============================================================================
# FILE PROCESSING WITH CONTEXT STITCHING
# =============================================================================

async def process_files(
    files,
    collection,
    file_list: List[dict],
    enable_context_stitching: bool = True,
    enable_layout_awareness: bool = True,
    enable_semantic_chunking: bool = True
):
    """
    Process uploaded files into ChromaDB with enhanced features.

    Features:
    - Semantic Chunking: Intelligent chunking based on semantic boundaries
    - Recursive Context Stitching: Bridges context between pages
    - Structural Layout Awareness: Preserves document structure
    - Multi-Stage OCR Verification: For images

    Args:
        files: List of uploaded files
        collection: ChromaDB collection
        file_list: List to track processed files
        enable_context_stitching: If True, extract and bridge page contexts
        enable_layout_awareness: If True, use layout-aware extraction
        enable_semantic_chunking: If True, use semantic-aware chunking (default)
    """

    progress = cl.Message(content="📂 Starting processing...")
    await progress.send()

    total_chunks = 0

    for i, file in enumerate(files):
        fname = file.name
        fpath = file.path

        progress.content = f"📂 Processing {i+1}/{len(files)}: {fname}"
        await progress.update()

        file_list.append({"name": fname})

        # PDF PROCESSING with enhancements
        if fname.lower().endswith('.pdf'):
            loop = asyncio.get_event_loop()

            # Initialize document context for context stitching
            doc_context = DocumentContext(filename=fname)

            # Use layout-aware extraction if enabled
            if enable_layout_awareness:
                progress.content = f"📐 Analyzing layout structure..."
                await progress.update()

                pages_with_layout = await loop.run_in_executor(
                    executor,
                    extract_pdf_pages_with_layout,
                    fpath,
                    True  # preserve_structure
                )

                progress.content = f"📄 Processing {len(pages_with_layout)} pages with layout awareness..."
                await progress.update()

                # Process pages with batched context stitching
                all_docs = []
                all_metas = []
                all_ids = []
                prev_context = None

                for batch_start in range(0, len(pages_with_layout), CONTEXT_BATCH_SIZE):
                    batch = pages_with_layout[batch_start:batch_start + CONTEXT_BATCH_SIZE]

                    # Batch extract contexts (skip short pages)
                    if enable_context_stitching:
                        eligible = [
                            (pn, pt) for pn, pt, _ in batch
                            if len(pt.strip()) >= MIN_PAGE_LENGTH
                        ]
                        if eligible:
                            try:
                                batch_contexts = await extract_batch_page_contexts(
                                    eligible, prev_context
                                )
                                for pc in batch_contexts:
                                    update_document_context(doc_context, pc)
                                if batch_contexts:
                                    prev_context = batch_contexts[-1]
                            except Exception as e:
                                print(f"⚠️ Batch context extraction error: {e}")

                    # Chunk all pages in this batch
                    for page_num, page_text, layout in batch:
                        context_bridge = ""
                        if enable_context_stitching and page_num > 1:
                            context_bridge = build_context_bridge(doc_context, page_num)

                        enriched_text = context_bridge + page_text if context_bridge else page_text

                        if enable_semantic_chunking:
                            chunks = hybrid_chunker.split_text(enriched_text, prefer_semantic=True)
                        else:
                            chunks = text_splitter.split_text(enriched_text)

                        for idx, chunk in enumerate(chunks):
                            all_docs.append(chunk)

                            meta = {
                                "page_number": page_num,
                                "source": fname,
                                "chunk_type": "text",
                                "layout_type": layout.layout_type,
                                "num_columns": layout.num_columns,
                                "has_tables": len(layout.tables) > 0
                            }

                            if enable_context_stitching and page_num in doc_context.page_contexts:
                                pc = doc_context.page_contexts[page_num]
                                meta["themes"] = ','.join(pc.themes[:3])
                                meta["key_entities"] = ','.join(pc.key_entities[:5])

                            all_metas.append(meta)
                            all_ids.append(
                                f"{fname}_p{page_num}_c{idx}_{uuid.uuid4().hex[:6]}"
                            )
            else:
                # Fallback to original extraction
                pages = await loop.run_in_executor(
                    executor,
                    extract_pdf_pages,
                    fpath
                )

                progress.content = f"📄 Chunking {len(pages)} pages..."
                await progress.update()

                all_docs = []
                all_metas = []
                all_ids = []
                prev_context = None

                for batch_start in range(0, len(pages), CONTEXT_BATCH_SIZE):
                    batch = pages[batch_start:batch_start + CONTEXT_BATCH_SIZE]

                    # Batch extract contexts (skip short pages)
                    if enable_context_stitching:
                        eligible = [
                            (pn, pt) for pn, pt in batch
                            if len(pt.strip()) >= MIN_PAGE_LENGTH
                        ]
                        if eligible:
                            try:
                                batch_contexts = await extract_batch_page_contexts(
                                    eligible, prev_context
                                )
                                for pc in batch_contexts:
                                    update_document_context(doc_context, pc)
                                if batch_contexts:
                                    prev_context = batch_contexts[-1]
                            except Exception as e:
                                print(f"⚠️ Batch context extraction error: {e}")

                    for page_num, page_text in batch:
                        context_bridge = ""
                        if enable_context_stitching and page_num > 1:
                            context_bridge = build_context_bridge(doc_context, page_num)

                        enriched_text = context_bridge + page_text if context_bridge else page_text

                        if enable_semantic_chunking:
                            chunks = hybrid_chunker.split_text(enriched_text, prefer_semantic=True)
                        else:
                            chunks = text_splitter.split_text(enriched_text)

                        for idx, chunk in enumerate(chunks):
                            all_docs.append(chunk)
                            all_metas.append({
                                "page_number": page_num,
                                "source": fname,
                                "chunk_type": "text"
                            })
                            all_ids.append(
                                f"{fname}_p{page_num}_c{idx}_{uuid.uuid4().hex[:6]}"
                            )

            print(f"📊 Created {len(all_docs)} chunks from {fname}")

            # Log context stitching stats
            if enable_context_stitching and doc_context.page_contexts:
                print(f"🔗 Context stitched across {len(doc_context.page_contexts)} pages")
                print(f"📌 Tracked {len(doc_context.global_entities)} entities, {len(doc_context.document_themes)} themes")

            # Embed and add in batches
            if all_docs:
                progress.content = f"⚡ Embedding {len(all_docs)} chunks..."
                await progress.update()

                all_embeddings = embed_documents(all_docs)

                add_chunks_to_collection(
                    collection,
                    all_docs,
                    all_embeddings,
                    all_metas,
                    all_ids
                )

                total_chunks += len(all_docs)

            # Process images with multi-stage verification
            progress.content = f"🖼️ Extracting images..."
            await progress.update()

            images = await loop.run_in_executor(
                executor,
                extract_pdf_images,
                fpath,
                25
            )

            if images:
                # Process images in parallel batches
                for batch_start in range(0, len(images), IMAGE_CONCURRENCY):
                    batch = images[batch_start:batch_start + IMAGE_CONCURRENCY]

                    progress.content = f"🖼️ Analyzing images {batch_start+1}-{batch_start+len(batch)}/{len(images)}..."
                    await progress.update()

                    # Launch batch in parallel
                    tasks = [
                        analyze_image(img, page_num, batch_start + j + 1, enable_verification=True)
                        for j, (page_num, img) in enumerate(batch)
                    ]
                    descriptions = await asyncio.gather(*tasks)

                    # Embed and store results
                    for j, ((page_num, img), desc) in enumerate(zip(batch, descriptions)):
                        img_embedding = embed_documents([desc])[0]

                        collection.add(
                            documents=[desc],
                            embeddings=[img_embedding],
                            metadatas=[{
                                "page_number": page_num,
                                "source": fname,
                                "chunk_type": "image"
                            }],
                            ids=[f"{fname}_img_p{page_num}_{batch_start+j}_{uuid.uuid4().hex[:6]}"]
                        )

                        total_chunks += 1
        
        # DOCX PROCESSING
        elif fname.lower().endswith('.docx'):
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(executor, extract_docx, fpath)

            if text:
                # Use semantic or character-based chunking
                if enable_semantic_chunking:
                    chunks = hybrid_chunker.split_text(text, prefer_semantic=True)
                else:
                    chunks = text_splitter.split_text(text)
                embeddings = embed_documents(chunks)
                
                docs = []
                metas = []
                ids = []
                
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    docs.append(chunk)
                    metas.append({
                        "source": fname,
                        "chunk_type": "text"
                    })
                    ids.append(f"{fname}_c{idx}_{uuid.uuid4().hex[:6]}")
                
                add_chunks_to_collection(
                    collection,
                    docs,
                    embeddings,
                    metas,
                    ids
                )
                
                total_chunks += len(chunks)
        
        # TXT PROCESSING
        elif fname.lower().endswith('.txt'):
            text = extract_txt(fpath)

            if text:
                # Use semantic or character-based chunking
                if enable_semantic_chunking:
                    chunks = hybrid_chunker.split_text(text, prefer_semantic=True)
                else:
                    chunks = text_splitter.split_text(text)
                embeddings = embed_documents(chunks)
                
                docs = []
                metas = []
                ids = []
                
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    docs.append(chunk)
                    metas.append({
                        "source": fname,
                        "chunk_type": "text"
                    })
                    ids.append(f"{fname}_c{idx}_{uuid.uuid4().hex[:6]}")
                
                add_chunks_to_collection(
                    collection,
                    docs,
                    embeddings,
                    metas,
                    ids
                )
                
                total_chunks += len(chunks)
        
        # IMAGE PROCESSING (original logic)
        elif fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            from PIL import Image
            img = Image.open(fpath)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            desc = await analyze_image(img, 0, 1)
            emb = embed_documents([desc])[0]
            
            collection.add(
                documents=[desc],
                embeddings=[emb],
                metadatas=[{
                    "source": fname,
                    "chunk_type": "image"
                }],
                ids=[f"{fname}_img_{uuid.uuid4().hex[:6]}"]
            )
            
            total_chunks += 1
    
    # Final summary with enhanced stats
    final_count = collection.count()

    # Get statistics
    try:
        all_meta = collection.get(include=["metadatas"])
        pages = [
            m.get("page_number")
            for m in all_meta["metadatas"]
            if m.get("page_number")
        ]
        page_range = f"{min(pages)}-{max(pages)}" if pages else "N/A"

        text_count = sum(
            1 for m in all_meta["metadatas"]
            if m.get("chunk_type") == "text"
        )
        img_count = sum(
            1 for m in all_meta["metadatas"]
            if m.get("chunk_type") == "image"
        )

        # Enhanced stats
        layout_aware_count = sum(
            1 for m in all_meta["metadatas"]
            if m.get("layout_type")
        )
        table_pages = sum(
            1 for m in all_meta["metadatas"]
            if m.get("has_tables")
        )
    except:
        page_range = "N/A"
        text_count = final_count
        img_count = 0
        layout_aware_count = 0
        table_pages = 0

    # Build enhanced summary message
    summary_parts = [
        f"✅ **Processing Complete!**\n",
        f"📊 **ChromaDB:**",
        f"• Total chunks: **{final_count}**",
        f"• Text: **{text_count}** | Images: **{img_count}**",
        f"• Pages: **{page_range}**",
        f"• Files: **{len(file_list)}**\n"
    ]

    # Add enhanced features info
    if enable_layout_awareness or enable_context_stitching or enable_semantic_chunking:
        summary_parts.append("🔬 **Enhanced Processing:**")
        if enable_semantic_chunking:
            summary_parts.append(f"• Semantic chunking: **Enabled**")
        if enable_layout_awareness and layout_aware_count > 0:
            summary_parts.append(f"• Layout-aware chunks: **{layout_aware_count}**")
            if table_pages > 0:
                summary_parts.append(f"• Pages with tables: **{table_pages}**")
        if enable_context_stitching:
            summary_parts.append(f"• Context stitching: **Enabled**")
        summary_parts.append("")

    summary_parts.extend([
        "💡 Try: 'What is on page 50?' or 'Summarize the document'\n",
        "Commands: `/clear`, `/files`, `/stats`, `/test`"
    ])

    await cl.Message(content="\n".join(summary_parts)).send()
