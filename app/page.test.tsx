import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import HomePage from './page'

vi.mock('@/lib/posts', () => ({
  getFeaturedPosts: () =>
    Array.from({ length: 4 }, (_, index) => ({
      slug: `post-${index + 1}`,
      title: `Post ${index + 1}`,
      date: `2026-01-0${index + 1}`,
      description: `Description ${index + 1}`,
      topics: ['Mathematics'],
      featured: true,
    })),
}))

describe('HomePage', () => {
  it('limits featured writing to three cards', () => {
    const html = renderToStaticMarkup(<HomePage />)

    expect(html.match(/<article/g)).toHaveLength(3)
    expect(html).not.toContain('Post 4')
  })
})
