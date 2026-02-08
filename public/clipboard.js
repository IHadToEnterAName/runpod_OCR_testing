// Clipboard fallback for non-HTTPS (HTTP) environments.
// The Clipboard API (navigator.clipboard.writeText) requires a secure context.
// This script provides a fallback using document.execCommand('copy').
(function () {
  if (window.isSecureContext) return;

  var fallbackCopy = function (text) {
    return new Promise(function (resolve, reject) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      textarea.style.top = "-9999px";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      try {
        var ok = document.execCommand("copy");
        document.body.removeChild(textarea);
        if (ok) {
          resolve();
        } else {
          reject(new Error("execCommand copy failed"));
        }
      } catch (err) {
        document.body.removeChild(textarea);
        reject(err);
      }
    });
  };

  if (
    !navigator.clipboard ||
    typeof navigator.clipboard.writeText !== "function"
  ) {
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: fallbackCopy,
        readText: function () {
          return Promise.resolve("");
        },
      },
      writable: true,
      configurable: true,
    });
  } else {
    navigator.clipboard.writeText = fallbackCopy;
  }
})();
