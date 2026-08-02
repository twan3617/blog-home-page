import { describe, expect, it } from 'vitest'
import LegacyPost from './page'

describe('legacy article redirect', () => {
  it('redirects encoded slugs without raw spaces or double encoding', async () => {
    const redirect = LegacyPost({
      params: Promise.resolve({
        slug: 'Bayesian Inference, and a basic Changepoint Detection Algorithm',
      }),
    })

    await expect(redirect).rejects.toMatchObject({
      digest: expect.stringContaining(
        '/writing/Bayesian%20Inference%2C%20and%20a%20basic%20Changepoint%20Detection%20Algorithm;308;',
      ),
    })
  })
})
