"""Check that nothing in report/report.html is clipped at A4 print width (needs an installed Chrome or Edge).

It lays the HTML out with print-media rules in a viewport as wide as an A4 page, puts the page margins back
as body padding, and reports any table, figure, image, SVG or code block that is wider than the printable
area, and any table whose content overflows its own box. Run after `python report/build_report.py`.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from cdp_browser import Browser, find_browser  # noqa: E402

A4_PX = 794  # 210 mm at 96 dpi
MARGIN_PX = round(15 * 96 / 25.4)  # the @page left and right margin (15 mm)

CHECK = """(() => {
  const limit = @A4@ - @MARGIN@ + 1.5, bad = [];
  for (const el of document.querySelectorAll('table, figure, svg, img, pre, .box, .toc')) {
    const r = el.getBoundingClientRect();
    if (r.right > limit + @MARGIN@ || r.left < @MARGIN@ - 1.5) bad.push([el.tagName, (el.tagName === 'TABLE' && el.caption ? el.caption.innerText.slice(0, 70) : '') || (el.className && el.className.baseVal === undefined ? el.className : '') || el.id || '', Math.round(r.left), Math.round(r.right)]);
    if (el.tagName === 'TABLE' && el.scrollWidth > el.clientWidth + 1) bad.push(['TABLE-OVERFLOW', el.className, el.scrollWidth, el.clientWidth]);
  }
  const wide = [...document.querySelectorAll('td, th')].filter(c => c.scrollWidth > c.clientWidth + 1).length;
  return {bad: bad.slice(0, 20), wideCells: wide, tables: document.querySelectorAll('table').length,
          figures: document.querySelectorAll('figure').length, scrollWidth: document.documentElement.scrollWidth};
})()""".replace("@A4@", str(A4_PX)).replace("@MARGIN@", str(MARGIN_PX))


def main() -> int:
    exe = find_browser()
    if exe is None:
        print("no Chrome or Edge installed: cannot check the layout")
        return 2
    html = ROOT / "report" / "report.html"
    browser = Browser(exe)
    try:
        page = browser.new_page()
        page.cmd("Emulation.setEmulatedMedia", {"media": "print"})
        page.viewport(A4_PX, 1123)
        page.goto(html.as_uri())
        page.js(
            f"document.body.style.padding = '0 {MARGIN_PX}px'; document.body.style.boxSizing = 'border-box'"
        )
        result = page.js(CHECK)
    finally:
        browser.close()
    print(json.dumps(result, indent=1))
    ok = not result["bad"] and result["scrollWidth"] <= A4_PX + 1
    print("LAYOUT OK: nothing wider than the printable area" if ok else "LAYOUT PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
