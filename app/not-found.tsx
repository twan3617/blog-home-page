import Link from 'next/link'

export default function NotFound() {
  return (
    <main className="notFound shell">
      <p className="archiveEyebrow">404 · Lost in the numbers</p>
      <h1>Page not found</h1>
      <p>
        This page may have moved, or the path may never have existed. There is
        still plenty to explore.
      </p>
      <nav className="notFoundActions" aria-label="Not found navigation">
        <Link className="button buttonPrimary" href="/">
          Return home
        </Link>
        <Link className="button buttonSecondary" href="/writing">
          Browse writing
        </Link>
      </nav>
    </main>
  )
}
