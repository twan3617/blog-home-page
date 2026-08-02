import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import ArticlePage, { generateMetadata, generateStaticParams } from './page'

describe('article route', () => {
  it('generates metadata and readable HTML for a Markdown article', async () => {
    const params = Promise.resolve({ slug: 'Borel-Cantelli' })
    const metadata = await generateMetadata({ params })
    const html = renderToStaticMarkup(await ArticlePage({ params }))

    expect(generateStaticParams()).toContainEqual({ slug: 'Borel-Cantelli' })
    expect(metadata.title).toBe('Infinitely Recurring Events occur with Probability Zero')
    expect(html).toContain('<article')
    expect(html).toContain('<h1')
    expect(html).toContain('class="katex"')
    expect(html).toContain('Back to all writing')
  })

  it('resolves URL-encoded slugs containing spaces', async () => {
    const slugs = [
      'Bayesian Inference, and a basic Changepoint Detection Algorithm',
      'Bayesian%20Inference%2C%20and%20a%20basic%20Changepoint%20Detection%20Algorithm',
    ]

    for (const slug of slugs) {
      const params = Promise.resolve({ slug })
      const metadata = await generateMetadata({ params })
      const html = renderToStaticMarkup(await ArticlePage({ params }))

      expect(metadata.title).toBe('Bayesian Inference, and a basic Changepoint Detection Algorithm')
      expect(html).toContain('Bayesian Inference, and a basic Changepoint Detection Algorithm')
    }
  })
})
