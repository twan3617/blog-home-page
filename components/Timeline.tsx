import Link from 'next/link'
import type { TimelineEntry } from '@/content/profile'
import styles from './Timeline.module.css'

export default function Timeline({ entries }: { entries: readonly TimelineEntry[] }) {
  return (
    <ol className={styles.timeline} role="list">
      {entries.map((entry) => (
        <li key={`${entry.period}-${entry.title}`} className={styles.entry}>
          <p className={styles.period}>{entry.period}</p>
          <h3>
            {entry.href ? <Link href={entry.href}>{entry.title}</Link> : entry.title}
          </h3>
          <p className={styles.subtitle}>{entry.subtitle}</p>
          {entry.location && <p className={styles.location}>{entry.location}</p>}
          <p className={styles.summary}>{entry.summary}</p>
          {entry.highlights && (
            <ul className={styles.highlights} aria-label="Highlights" role="list">
              {entry.highlights.map((highlight) => (
                <li key={highlight}>{highlight}</li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ol>
  )
}
