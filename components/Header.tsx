import Link from 'next/link'
import { profile } from '@/content/profile'
import styles from './Header.module.css'

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={`${styles.inner} shell`}>
        <Link className={styles.name} href="/" aria-label="Tony Wang, home">
          TW
        </Link>
        <nav className={styles.navigation} aria-label="Primary navigation">
          {profile.navigation.map((item) => (
            <Link key={item.href} href={`/${item.href}`}>
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  )
}
