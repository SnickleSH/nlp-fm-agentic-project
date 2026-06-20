// Markdown -> standalone HTML for PDF export (see scripts/build_report.sh).
// Renders KaTeX math at build time and INLINES the KaTeX stylesheet with its
// woff2 fonts base64-embedded, so headless-Chrome printing never races a CDN
// (the cause of broken sub/superscripts and fractions in earlier exports).
// Deps (markdown-it, markdown-it-katex, katex) resolve from $REPORT_BUILD_DEPS.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'module';

const depsDir = process.env.REPORT_BUILD_DEPS || process.cwd();
const require = createRequire(path.join(depsDir, 'package.json'));
const MarkdownIt = require('markdown-it');
const katex = require('markdown-it-katex');

const [, , mdPath, htmlOut] = process.argv;
if (!mdPath || !htmlOut) {
  console.error('usage: node render_report.mjs <input.md> <output.html>');
  process.exit(1);
}

// --- KaTeX stylesheet with fonts inlined as base64 woff2 ---------------------
function inlineKatexCss() {
  const katexDist = path.join(depsDir, 'node_modules', 'katex', 'dist');
  let css = fs.readFileSync(path.join(katexDist, 'katex.min.css'), 'utf8');
  // Embed each woff2; drop the woff/ttf fallbacks (they'd reference missing files).
  css = css.replace(/url\(fonts\/([^)]+\.woff2)\)/g, (_m, file) => {
    const b64 = fs.readFileSync(path.join(katexDist, 'fonts', file)).toString('base64');
    return `url(data:font/woff2;base64,${b64})`;
  });
  css = css.replace(/,\s*url\(fonts\/[^)]+\)\s*format\("(?:woff|truetype)"\)/g, '');
  return css;
}

const src = fs.readFileSync(mdPath, 'utf8');
const md = new MarkdownIt({ html: true, linkify: true, typographer: false }).use(katex);
const body = md.render(src);

const pageCss = `
@page { size: A4; margin: 16mm 18mm; }
* { box-sizing: border-box; }
body { font-family: Georgia,"Times New Roman",serif; font-size: 10.5pt; line-height: 1.38; color:#111; }
h1 { font-size: 18pt; line-height: 1.2; margin: 0 0 .3em; text-align: center; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin: 1em 0 .4em; }
h3 { font-size: 11.5pt; margin: .8em 0 .3em; }
h4 { font-size: 10.8pt; margin: .7em 0 .2em; }
p  { margin: .45em 0; text-align: left; }
ol, ul { margin: .4em 0; padding-left: 1.4em; }
li { margin: .12em 0; }
[align="center"] { text-align: center !important; }
code { font-family: "DejaVu Sans Mono",monospace; font-size: 9pt; background:#f4f4f4; padding:0 2px; border-radius:3px; }
pre { background:#f4f4f4; padding:7px 9px; border-radius:5px; overflow-x:auto; }
pre code { background:none; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: .5em 0; }
th, td { border: 1px solid #bbb; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background:#f0f0f0; }
img { max-width: 100%; height: auto; display: block; margin: .3em auto; }
blockquote { border-left: 3px solid #ccc; margin: .5em 0; padding: .1em 0 .1em 12px; color:#444; }
hr { border:none; border-top:1px solid #ddd; margin: .9em 0; }
.katex-display { margin: .5em 0; overflow-x: auto; overflow-y: hidden; }
`;

const html = `<!doctype html><html><head><meta charset="utf-8">
<style>${inlineKatexCss()}</style>
<style>${pageCss}</style></head><body>${body}</body></html>`;

fs.writeFileSync(htmlOut, html);
console.error(`rendered ${path.basename(mdPath)} -> ${path.basename(htmlOut)} (${(html.length/1024).toFixed(0)} KB)`);
