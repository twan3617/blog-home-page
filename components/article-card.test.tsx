import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import ArticleCard from './ArticleCard'

describe('ArticleCard', () => {
  it('renders complete article metadata', () => {
    const html = renderToStaticMarkup(
      <ArticleCard
        post={{
          slug: 'random walk',
          title: 'Random Walks',
          date: '2022-04-01',
          description: 'A probabilistic journey home.',
          topics: ['Probability', 'Computation'],
          featured: true,
        }}
      />,
    )

    expect(html).toContain('Random Walks')
    expect(html).toContain('dateTime="2022-04-01"')
    expect(html).toContain('A probabilistic journey home.')
    expect(html).toContain('Probability')
    expect(html).toContain('Computation')
    expect(html).toContain('/writing/random%20walk')
  })
})
