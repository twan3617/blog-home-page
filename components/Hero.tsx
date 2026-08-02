import Image from 'next/image'
import Link from 'next/link'
import { profile } from '@/content/profile'
import styles from './Hero.module.css'

export default function Hero() {
  return (
    <section className={`${styles.hero} shell`} aria-labelledby="hero-title">
      <div className={styles.copy}>
        <p className={styles.eyebrow}>Mathematics · Computation · Writing</p>
        <h1 id="hero-title">{profile.name}</h1>
        <p className={styles.statement}>{profile.statement}</p>
        <p className={styles.introduction}>{profile.introduction}</p>
        <div className={styles.actions}>
          <a className="button buttonPrimary" href="#writing">Read my writing</a>
          <a className="button buttonSecondary" href="#about">About me</a>
          <Link className={styles.resume} href={profile.links.resume}>Résumé ↗</Link>
        </div>
      </div>
      <div className={styles.portraitFrame}>
        <Image
          src="/images/profile.jpeg"
          alt="Tony Wang"
          width={640}
          height={640}
          priority
          sizes="(max-width: 48rem) 82vw, 34rem"
          className={styles.portrait}
        />
      </div>
    </section>
  )
}
