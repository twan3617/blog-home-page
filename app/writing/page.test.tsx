import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import WritingPage, { metadata } from './page'

describe('WritingPage', () => {
  it('renders the complete archive with a logical heading hierarchy', () => {
    const html = renderToStaticMarkup(<WritingPage />)

    expect(html.match(/<article/g)).toHaveLength(6)
    expect(html).toMatch(/<h1[^>]*>Writing<\/h1>[\s\S]*<h2[^>]*>Article archive<\/h2>[\s\S]*<h3/)
  })

  it('publishes canonical social metadata', () => {
    expect(metadata).toMatchObject({
      alternates: { canonical: '/writing' },
      openGraph: {
        url: '/writing',
        siteName: 'Tony Wang',
        locale: 'en_GB',
        images: [
          {
            url: '/opengraph-image',
            width: 1200,
            height: 630,
          },
        ],
      },
      twitter: {
        card: 'summary_large_image',
        images: ['/opengraph-image'],
      },
    })
  })
})
