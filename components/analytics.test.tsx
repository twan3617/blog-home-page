import type { ScriptProps } from 'next/script'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, expect, it, vi } from 'vitest'
import Analytics from './Analytics'

// Expose the script configuration without fetching a third-party script in Node.
vi.mock('next/script', () => ({
  default: (props: ScriptProps & Record<string, unknown>) => (
    <script
      async
      src={props.src}
      data-goatcounter={props['data-goatcounter'] as string}
      data-goatcounter-settings={props['data-goatcounter-settings'] as string}
    />
  ),
}))

afterEach(() => vi.unstubAllEnvs())

it('loads no analytics script without an endpoint', () => {
  vi.stubEnv('NEXT_PUBLIC_GOATCOUNTER_URL', '')
  expect(renderToStaticMarkup(<Analytics />)).toBe('')
})

it('configures manual page counting so loading the script cannot double-count a visit', () => {
  vi.stubEnv('NEXT_PUBLIC_GOATCOUNTER_URL', 'https://example.goatcounter.com/count')
  const html = renderToStaticMarkup(<Analytics />)
  expect(html).toContain('src="https://gc.zgo.at/count.js"')
  expect(html).toContain('data-goatcounter="https://example.goatcounter.com/count"')
  expect(html).toContain('data-goatcounter-settings="{&quot;no_onload&quot;:true}"')
})
