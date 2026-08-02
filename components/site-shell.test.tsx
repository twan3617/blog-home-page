import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import Header from './Header'
import Hero from './Hero'
import Footer from './Footer'

describe('site shell', () => {
  it('renders semantic navigation and identity', () => {
    const html = renderToStaticMarkup(
      <>
        <Header />
        <Hero />
        <Footer />
      </>,
    )

    expect(html.match(/<nav/g)).toHaveLength(1)
    expect(html.match(/<h1/g)).toHaveLength(1)
    expect(html).toContain('alt="Tony Wang"')
    expect(html).toContain('LinkedIn')
    expect(html).toContain('GitHub')
    expect(html).toContain('Email')
  })
})
