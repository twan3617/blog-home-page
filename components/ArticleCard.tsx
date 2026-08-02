import Link from 'next/link'
import type { PostSummary } from '@/lib/posts'
import styles from './ArticleCard.module.css'

const formatter = new Intl.DateTimeFormat('en-GB', {
  dateStyle: 'long',
  timeZone: 'UTC',
})

export default function ArticleCard({ post }: { post: PostSummary }) {
  return (
    <article className={styles.card}>
      <time className={styles.date} dateTime={post.date}>
        {formatter.format(new Date(post.date))}
      </time>
      <h3>
        <Link href={`/writing/${encodeURIComponent(post.slug)}`}>{post.title}</Link>
      </h3>
      <p>{post.description}</p>
      <ul className={styles.topics} aria-label="Topics">
        {post.topics.map((topic) => (
          <li key={topic}>{topic}</li>
        ))}
      </ul>
    </article>
  )
}
