import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import WritingPage from './page'

describe('WritingPage', () => {
  it('renders the complete archive with a logical heading hierarchy', () => {
    const html = renderToStaticMarkup(<WritingPage />)

    expect(html.match(/<article/g)).toHaveLength(6)
    expect(html).toMatch(/<h1[^>]*>Writing<\/h1>[\s\S]*<h2[^>]*>Article archive<\/h2>[\s\S]*<h3/)
  })
})
