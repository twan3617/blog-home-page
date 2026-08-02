import Link from 'next/link'
import { profile } from '@/content/profile'
import styles from './Footer.module.css'

export default function Footer() {
  return (
    <footer id="contact" className={styles.footer}>
      <div className={`${styles.inner} shell`}>
        <div>
          <p className={styles.eyebrow}>Let’s talk</p>
          <h2>Have an interesting problem or idea?</h2>
          <a className={styles.email} href={profile.links.email}>Email me ↗</a>
        </div>
        <div className={styles.links}>
          <a href={profile.links.linkedin}>LinkedIn</a>
          <a href={profile.links.github}>GitHub</a>
          <a href={profile.links.email}>Email</a>
          <Link href={profile.links.resume}>Résumé</Link>
        </div>
        <p className={styles.note}>© {new Date().getFullYear()} Tony Wang · {profile.location}</p>
      </div>
    </footer>
  )
}
