import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { getPost, getPostSlugs, readPosts } from './posts'

describe('readPosts', () => {
  const directories: string[] = []

  afterEach(() => {
    directories.splice(0).forEach((directory) => {
      rmSync(directory, { recursive: true, force: true })
    })
  })

  it('sorts valid posts newest first and exposes typed metadata', () => {
    const directory = mkdtempSync(join(tmpdir(), 'posts-'))
    directories.push(directory)
    writeFileSync(join(directory, 'older.md'), `---
title: Older
date: 2025-01-01
description: Older post
topics: [Math]
featured: false
---
Body`)
    writeFileSync(join(directory, 'newer.md'), `---
title: Newer
date: 2026-01-01
description: Newer post
topics: [Code]
featured: true
---
Body`)

    expect(readPosts(directory).map(({ slug }) => slug)).toEqual(['newer', 'older'])
  })

  it('reports the file when required metadata is missing', () => {
    const directory = mkdtempSync(join(tmpdir(), 'posts-'))
    directories.push(directory)
    writeFileSync(join(directory, 'broken.md'), `---
title: Broken
date: 2026-01-01
---
Body`)

    expect(() => readPosts(directory)).toThrow(
      'broken.md: description must be a non-empty string',
    )
  })
})

describe('getPost', () => {
  it('renders mathematics and trusted local HTML', async () => {
    const post = await getPost('Borel-Cantelli')

    expect(post.contentHtml).toContain('class="katex"')
    expect(post.contentHtml).toContain('<br>')
  })

  it('renders every article without KaTeX warnings or errors', async () => {
    const warnings: string[] = []
    const warn = vi.spyOn(console, 'warn').mockImplementation((message) => {
      warnings.push(String(message))
    })

    try {
      for (const slug of getPostSlugs()) {
        const post = await getPost(slug)
        expect(post.contentHtml).not.toContain('katex-error')
      }
    } finally {
      warn.mockRestore()
    }

    expect(warnings).toEqual([])
  })

  it('keeps article headings below the page title and describes every figure', async () => {
    for (const slug of getPostSlugs()) {
      const { contentHtml } = await getPost(slug)
      expect(contentHtml).not.toMatch(/<h1\b/)
      for (const [, alt] of contentHtml.matchAll(/<img\b[^>]*alt="([^"]*)"/g)) {
        expect(alt).not.toMatch(/^(?:TOP_PHOTO|drawing\d+|diagnostics charts)$/i)
        expect(alt.length).toBeGreaterThan(10)
      }
    }
  })
})
