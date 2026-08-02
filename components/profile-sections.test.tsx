import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { profile } from '@/content/profile'
import InterestList from './InterestList'
import Timeline from './Timeline'

describe('profile sections', () => {
  it('renders ordered experience and linked education', () => {
    const experience = renderToStaticMarkup(<Timeline entries={profile.experience} />)
    const education = renderToStaticMarkup(<Timeline entries={profile.education} />)

    expect(experience).toContain('<ol')
    expect(experience.indexOf('Coolabah')).toBeLessThan(experience.indexOf('Quantium'))
    expect(experience).toContain('2025 — Present')
    expect(education).toContain('/assets/thesis/masters_thesis_tw.pdf')
  })

  it('renders current and personal interests as accessible lists', () => {
    const current = renderToStaticMarkup(<InterestList items={profile.current} />)
    const interests = renderToStaticMarkup(<InterestList items={profile.interests} />)

    expect(current).toContain('<ul')
    expect(current).toContain('role="list"')
    expect(current).toContain('Coolabah Capital Investments')
    expect(interests).toContain('Volleyball')
  })
})
