(function () {
  "use strict";

  var primaryTile = document.getElementById("primaryTile");
  var secondaryTile = document.getElementById("secondaryTile");
  var primaryInput = document.getElementById("primaryInput");
  var secondaryInput = document.getElementById("secondaryInput");
  var primaryFileName = document.getElementById("primaryFileName");
  var secondaryFileName = document.getElementById("secondaryFileName");
  var runBtn = document.getElementById("runBtn");
  var pathLoading = document.getElementById("pathLoading");
  var runHint = document.getElementById("runHint");
  var successMsg = document.getElementById("successMsg");
  var errorMsg = document.getElementById("errorMsg");
  var pdfTile = document.getElementById("pdfTile");
  var wordTile = document.getElementById("wordTile");

  var primaryFile = null;
  var secondaryFile = null;

  function setError(text) {
    if (text) {
      errorMsg.textContent = text;
      errorMsg.hidden = false;
    } else {
      errorMsg.textContent = "";
      errorMsg.hidden = true;
    }
  }

  function setSuccess(text) {
    if (text) {
      successMsg.textContent = text;
      successMsg.hidden = false;
    } else {
      successMsg.textContent = "";
      successMsg.hidden = true;
    }
  }

  function updateRunState() {
    var ready = !!(primaryFile && secondaryFile);
    runBtn.disabled = !ready;
    if (runHint) {
      runHint.hidden = ready;
    }
  }

  function setVisualTileKind(kind) {
    var label = pdfTile && pdfTile.querySelector(".tile-label");
    if (!label) return;
    label.textContent =
      kind === "png" ? "Annotated comparison image" : "Annotated comparison PDF";
  }

  function resetOutputTiles() {
    setSuccess("");
    setVisualTileKind("pdf");
    [pdfTile, wordTile].forEach(function (el) {
      el.removeAttribute("href");
      el.removeAttribute("download");
      el.removeAttribute("target");
      el.removeAttribute("rel");
      el.setAttribute("aria-disabled", "true");
      el.tabIndex = -1;
      var hint = el.querySelector(".tile-hint");
      if (hint) {
        hint.removeAttribute("aria-hidden");
        hint.textContent = "Not available yet";
        hint.setAttribute("data-empty", "");
      }
    });
  }

  /**
   * @param {HTMLElement} tile
   * @param {string | null | undefined} href
   * @param {{ openInNewTab?: boolean }} [opts]
   */
  function applyOutputHref(tile, href, opts) {
    opts = opts || {};
    var hint = tile.querySelector(".tile-hint");
    if (!href || typeof href !== "string" || !href.trim()) {
      tile.removeAttribute("href");
      tile.removeAttribute("download");
      tile.removeAttribute("target");
      tile.removeAttribute("rel");
      tile.setAttribute("aria-disabled", "true");
      tile.tabIndex = -1;
      if (hint) {
        hint.textContent = "Not available yet";
        hint.setAttribute("data-empty", "");
      }
      return;
    }
    tile.href = href.trim();
    tile.setAttribute("aria-disabled", "false");
    tile.tabIndex = 0;
    if (opts.openInNewTab) {
      tile.target = "_blank";
      tile.rel = "noopener noreferrer";
      tile.removeAttribute("download");
    } else {
      tile.removeAttribute("target");
      tile.removeAttribute("rel");
      tile.setAttribute("download", "");
    }
    if (hint) {
      hint.textContent = opts.openInNewTab ? "Open in new tab" : "Download";
      hint.removeAttribute("data-empty");
    }
  }

  function bindInputTile(tile, input, labelEl, setFile) {
    tile.addEventListener("click", function () {
      input.click();
    });
    input.addEventListener("change", function () {
      var f = input.files && input.files[0];
      setFile(f || null);
      labelEl.textContent = f ? f.name : "";
      updateRunState();
    });

    ["dragenter", "dragover"].forEach(function (ev) {
      tile.addEventListener(ev, function (e) {
        e.preventDefault();
        e.stopPropagation();
        tile.classList.add("dragover");
      });
    });
    tile.addEventListener("dragleave", function (e) {
      e.preventDefault();
      tile.classList.remove("dragover");
    });
    tile.addEventListener("drop", function (e) {
      e.preventDefault();
      e.stopPropagation();
      tile.classList.remove("dragover");
      var dt = e.dataTransfer;
      if (!dt || !dt.files || !dt.files.length) return;
      var f = dt.files[0];
      setFile(f);
      labelEl.textContent = f.name;
      updateRunState();
    });
  }

  bindInputTile(primaryTile, primaryInput, primaryFileName, function (f) {
    primaryFile = f;
  });
  bindInputTile(secondaryTile, secondaryInput, secondaryFileName, function (f) {
    secondaryFile = f;
  });

  function parseErrorDetail(detail) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map(function (d) {
          return typeof d === "object" && d && d.msg ? String(d.msg) : String(d);
        })
        .join("; ");
    }
    return "";
  }

  function parseResponseBody(res, text) {
    var ct = res.headers.get("content-type") || "";
    if (ct.indexOf("application/json") !== -1) {
      try {
        return JSON.parse(text);
      } catch (e) {
        return null;
      }
    }
    return null;
  }

  runBtn.addEventListener("click", function () {
    if (!primaryFile || !secondaryFile) return;
    setError("");
    resetOutputTiles();
    pathLoading.hidden = false;
    runBtn.disabled = true;

    var form = new FormData();
    form.append("source_file", primaryFile, primaryFile.name);
    form.append("target_file", secondaryFile, secondaryFile.name);
    form.append("include_text_reports", "false");

    fetch("/compare", { method: "POST", body: form })
      .then(function (res) {
        return res.text().then(function (text) {
          var body = parseResponseBody(res, text);
          return { ok: res.ok, status: res.status, body: body, rawText: text };
        });
      })
      .then(function (result) {
        pathLoading.hidden = true;
        runBtn.disabled = false;
        updateRunState();

        if (!result.ok) {
          var d = result.body && result.body.detail;
          var msg = parseErrorDetail(d);
          if (!msg && result.rawText && result.rawText.length < 500) {
            msg = result.rawText.trim();
          }
          setError(msg || "Request failed (" + result.status + ")");
          return;
        }
        if (!result.body) {
          setError("Invalid response");
          return;
        }
        var extras = result.body.extras;
        if (extras && typeof extras === "object") {
          var visualHref =
            extras.output_visual_href || extras.output_pdf_href || extras.output_png_href;
          var visualKind = extras.output_visual_kind || (extras.output_png_href ? "png" : "pdf");
          setVisualTileKind(visualKind);
          applyOutputHref(pdfTile, visualHref);
          applyOutputHref(wordTile, extras.output_word_href);
        }
        setSuccess(
          "Comparison finished — download the annotated comparison and the Word change summary.",
        );
      })
      .catch(function () {
        pathLoading.hidden = true;
        runBtn.disabled = false;
        updateRunState();
        setError("Network error");
      });
  });

  updateRunState();
})();
