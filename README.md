# Mathematics, Statistics and Everything Else - Tony Wang

Hi! 

I am a data scientist based in Sydney who likes to write and share things online.

This is the repository behind my [Mathematics, Statistics and Everything Else blog](https://maths-stats-and-everything-else.netlify.app/.), currently being hosted on Netlify.

Please visit the website to get contact details!

## About

This blog builds on [Next.js](https://nextjs.org/learn/basics/create-nextjs-app?utm_source=next-site&utm_medium=nav-cta&utm_campaign=next-website)'s TypeScript starter blog with added [KaTeX](https://github.com/KaTeX/KaTeX) support in markdown files.

It uses the [UnifiedJS](https://unifiedjs.com) ecosystem to parse and process markdown files located in `posts/`. An example post is included: `posts/markdown-math.md`.

Use `$` (inline) or `$$` (blocks) to wrap KaTeX syntax in your files. This marks them for processing by `remark-math`.

All text processing takes place in `lib/posts.ts`, utilizing a handful of [Remark](https://remark.js.org) plugins. [Remark-math](https://github.com/remarkjs/remark-math) parses `$` and `$$` into math nodes. [Remark-html-katex](https://github.com/remarkjs/remark-math/tree/main/packages/remark-html-katex) transforms those nodes with KaTeX. The necessary css for rendering KaTeX is imported in `_app.tsx`.

Credits to [Abigail Young](https://abigailyoung.com/) for the original code to get the math processing working!

## Getting Started

```sh
yarn install
yarn dev
```

This should get you up and running. Open in your browser at `localhost:3000`.

To view KaTeX in action, click on the blog post titled `KaTeX and Markdown`. If everything works properly, you should see the following rendered:

<img src="katex-example.png" width="500" alt="KaTeX example">. 



