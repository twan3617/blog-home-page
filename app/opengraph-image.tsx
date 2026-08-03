import { ImageResponse } from 'next/og'
import { site } from '@/lib/site'

export const alt = site.socialImage.alt
export const size = {
  width: site.socialImage.width,
  height: site.socialImage.height,
}
export const contentType = 'image/png'

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          position: 'relative',
          display: 'flex',
          width: '100%',
          height: '100%',
          flexDirection: 'column',
          justifyContent: 'space-between',
          overflow: 'hidden',
          padding: '72px 84px',
          color: '#14272d',
          backgroundColor: '#f4f0e7',
          backgroundImage:
            'linear-gradient(rgba(57,119,122,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(57,119,122,0.08) 1px, transparent 1px)',
          backgroundSize: '64px 64px',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: '-140px',
            right: '-80px',
            display: 'flex',
            width: '520px',
            height: '520px',
            borderRadius: '50%',
            background: 'rgba(227,180,119,0.32)',
          }}
        />
        <div
          style={{
            display: 'flex',
            color: '#25575b',
            fontSize: 24,
            fontWeight: 700,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
          }}
        >
          Mathematics · Computation · Writing
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div
            style={{
              display: 'flex',
              fontFamily: 'Georgia, serif',
              fontSize: 112,
              fontWeight: 600,
              letterSpacing: '-0.06em',
              lineHeight: 0.9,
            }}
          >
            Tony Wang
          </div>
          <div
            style={{
              display: 'flex',
              maxWidth: '900px',
              marginTop: '36px',
              color: '#53636a',
              fontSize: 34,
              lineHeight: 1.3,
            }}
          >
            Exploring mathematics, computation, and the systems they help us
            understand.
          </div>
        </div>
        <div style={{ display: 'flex', color: '#a9692e', fontSize: 24 }}>
          Quantitative finance · London
        </div>
      </div>
    ),
    size,
  )
}
