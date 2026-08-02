import type { ReactNode } from 'react'
import styles from './Section.module.css'

type SectionProps = {
  id: string
  eyebrow?: string
  title: string
  children: ReactNode
}

export default function Section({ id, eyebrow, title, children }: SectionProps) {
  return (
    <section id={id} className={`${styles.section} shell reveal`}>
      <div className={styles.heading}>
        {eyebrow && <p className={styles.eyebrow}>{eyebrow}</p>}
        <h2>{title}</h2>
      </div>
      <div className={styles.content}>{children}</div>
    </section>
  )
}
