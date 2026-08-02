import { describe, expect, it } from 'vitest'
import { profile } from './profile'

describe('profile content', () => {
  it('keeps the current role first and all external links absolute', () => {
    expect(profile.experience[0]).toMatchObject({
      subtitle: 'Coolabah Capital Investments',
      location: 'London, United Kingdom',
    })

    for (const href of [profile.links.linkedin, profile.links.github]) {
      expect(href).toMatch(/^https:\/\//)
    }
  })
})
