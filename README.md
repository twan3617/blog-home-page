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

Most components are Server Components. The optional analytics component runs in the browser to track page changes; content rendering, responsive layout, and progressive animation are handled by HTML and CSS. This keeps the browser JavaScript footprint small.

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

The production site is hosted on Netlify. `netlify.toml` sets the build command to `npm run build` and the publish directory to `.next`; `.nvmrc` records Node `24.18.1`. Next.js and Netlify handle the statically generated pages and compatibility redirects from the committed App Router source.

## Visit analytics

The site supports [GoatCounter](https://www.goatcounter.com/) for visits per page and blog post. Tracking stays disabled when its environment variable is unset.

1. Create a GoatCounter site and copy the `data-goatcounter` URL from its tracking snippet.
2. In Netlify's project environment variables, add `NEXT_PUBLIC_GOATCOUNTER_URL` with that full URL, for example `https://YOUR_CODE.goatcounter.com/count`. Leave **Contains secret values** unchecked. Make it available to **Builds** and set its value for the **Production** deploy context only, leaving previews unset.
3. Trigger a new production build and deploy. Next.js embeds `NEXT_PUBLIC_` values at build time, so changing this variable always requires a rebuild. The URL is public configuration, not an API key.

`netlify.toml` excludes this public URL from Netlify's secret scanning, including if the variable was previously marked as secret. Other variables remain subject to scanning.

The shared layout loads the script once and counts the initial page plus subsequent Next.js page changes. These are visits, not proof that someone finished reading an article. GoatCounter starts collecting after you enable it; it does not recover earlier visits.

To [exclude your own browser](https://www.goatcounter.com/help/skip-dev), load `https://maths-stats-and-everything-else.netlify.app/#toggle-goatcounter` and confirm the alert says tracking is disabled. Reload if needed, then remove the fragment from the URL. Repeat in each browser/device you use; opening that special URL again toggles tracking back on. Localhost visits are ignored automatically.

To check the integration after deployment, use a browser without the exclusion or an analytics blocker. Open the homepage, click through to Writing and a post, and check the browser Network panel for one request to your GoatCounter `/count` endpoint per page, with the visited path in the `p` query parameter. The script should load once, and navigation should still work if you block `gc.zgo.at`.
