import type { InterestGroup } from '@/content/profile'
import styles from './InterestList.module.css'

export default function InterestList({ items }: { items: readonly InterestGroup[] }) {
  return (
    <ul className={styles.list} role="list">
      {items.map((item) => (
        <li key={item.title}>
          <h3>{item.title}</h3>
          <p>{item.description}</p>
        </li>
      ))}
    </ul>
  )
}
