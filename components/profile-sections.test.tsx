import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { InterestGroup, TimelineEntry } from '@/content/profile'
import InterestList from './InterestList'
import Timeline from './Timeline'

describe('profile sections', () => {
  it('renders ordered experience and linked education', () => {
    const entries: readonly TimelineEntry[] = [
      {
        period: 'Later',
        title: 'Linked entry',
        subtitle: 'Organisation',
        summary: 'A linked timeline entry.',
        href: '/document.pdf',
      },
      {
        period: 'Earlier',
        title: 'Earlier entry',
        subtitle: 'Organisation',
        summary: 'An earlier timeline entry.',
      },
    ]
    const timeline = renderToStaticMarkup(<Timeline entries={entries} />)

    expect(timeline).toContain('<ol')
    expect(timeline.indexOf('Linked entry')).toBeLessThan(timeline.indexOf('Earlier entry'))
    expect(timeline).toContain('/document.pdf')
  })

  it('renders current and personal interests as accessible lists', () => {
    const items: readonly InterestGroup[] = [
      { title: 'An interest', description: 'A short description.' },
    ]
    const interests = renderToStaticMarkup(<InterestList items={items} />)

    expect(interests).toContain('<ul')
    expect(interests).toContain('role="list"')
    expect(interests).toContain('An interest')
    expect(interests).toContain('A short description.')
  })
})
