'use client'

import Script from 'next/script'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'

declare global {
  interface Window {
    goatcounter?: {
      count: (visit: { path: string }) => void
    }
  }
}

export default function Analytics() {
  const endpoint = process.env.NEXT_PUBLIC_GOATCOUNTER_URL?.trim()
  const pathname = usePathname()
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (endpoint && ready && pathname) {
      window.goatcounter?.count({ path: pathname })
    }
  }, [endpoint, pathname, ready])

  if (!endpoint) return null

  return (
    <Script
      src="https://gc.zgo.at/count.js"
      data-goatcounter={endpoint}
      data-goatcounter-settings='{"no_onload":true}'
      onReady={() => setReady(true)}
    />
  )
}
