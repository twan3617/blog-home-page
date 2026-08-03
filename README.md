# Tony Wang — personal website

The source for [maths-stats-and-everything-else.netlify.app](https://maths-stats-and-everything-else.netlify.app/), a personal site about mathematics, computation, writing, and quantitative finance.

The site uses Next.js 16 and the App Router. Pages are statically generated wherever possible, including the homepage, writing archive, Markdown articles, sitemap, robots file, and social sharing image.

## Local development

The required Node version is recorded in `.node-version`, `.nvmrc`, and `package.json`:

```sh
nodenv install 24.18.1 # only needed once
nodenv local 24.18.1
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Before committing a substantial change, run:

```sh
npm test
npm run lint
npm run typecheck
npm run build
```

`npm ci` installs exactly the dependency versions recorded in `package-lock.json`. Use npm rather than maintaining a second lockfile for another package manager.

## Architecture

```text
content/profile.ts          Personal, professional, and navigation content
posts/*.md                  Long-form writing and article metadata
lib/posts.ts                Frontmatter validation and Markdown-to-HTML pipeline
lib/site.ts                 Canonical URL and shared site metadata
app/                        App Router pages, layouts, metadata, and generated routes
components/                 Reusable server-rendered presentation components
public/                     Portraits, article images, résumé, and theses
```

The content flow is:

```text
Markdown or typed profile data
  → validation and transformation
  → reusable React components
  → statically generated HTML
```

Most components are Server Components. The site currently needs no application-level Client Components because navigation, content rendering, responsive layout, and progressive animation are handled by HTML and CSS. This keeps the browser JavaScript footprint small.

Global design tokens and shared layout utilities live in `app/globals.css`. Component-specific styles use CSS Modules, keeping selectors local to the component that imports them.

## Adding an article

Create a Markdown file in `posts/` with this frontmatter:

```yaml
---
title: "Article title"
date: "2026-08-03"
description: "A concise summary for cards and search metadata."
topics: [Mathematics, Computation]
featured: false
---
```

The filename becomes the article slug. The writing archive, static article route, metadata, and sitemap are all generated from the same source. Set `featured: true` to make the article eligible for the three-card homepage selection.

Inline mathematics uses `$...$`; display mathematics uses `$$...$$`. The Markdown pipeline uses Unified, Remark, Rehype, and KaTeX. Raw HTML is supported because the repository content is trusted local input; it should not be enabled unchanged for user-submitted content.

## Updating profile content

Edit `content/profile.ts` to change the introduction, current interests, experience, education, personal interests, navigation, or external links. The components consume typed data rather than embedding separate copies of this information in the page markup.

## Routes and discoverability

- `/` — personal homepage
- `/writing` — complete writing archive
- `/writing/[slug]` — statically generated article
- `/posts/[slug]` — permanent compatibility redirect for old links
- `/sitemap.xml` and `/robots.txt` — generated crawler routes
- `/opengraph-image` — generated 1200×630 sharing image

Next.js metadata exports generate page titles, descriptions, canonical URLs, and Open Graph/Twitter tags on the server. Missing routes use the custom `app/not-found.tsx` page.

## Deployment

The production site is hosted on Netlify. Configure the deployment to use Node `24.18.1`, install with `npm ci`, and build with `npm run build`. Next.js and Netlify handle the statically generated pages and compatibility redirects from the committed App Router source.
