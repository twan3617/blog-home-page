import { describe, expect, it } from 'vitest'
import { getPostSlugs } from '@/lib/posts'
import sitemap from './sitemap'

describe('sitemap', () => {
  it('contains canonical public routes only', () => {
    const urls = sitemap().map(({ url }) => url)

    expect(urls).toContain('https://maths-stats-and-everything-else.netlify.app/')
    expect(urls).toContain(
      'https://maths-stats-and-everything-else.netlify.app/writing',
    )
    for (const slug of getPostSlugs()) {
      expect(urls).toContain(
        `https://maths-stats-and-everything-else.netlify.app/writing/${encodeURIComponent(slug)}`,
      )
    }
    expect(urls.some((url) => url.includes('/posts/'))).toBe(false)
  })
})
