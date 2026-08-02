export type TimelineEntry = {
  period: string
  title: string
  subtitle: string
  location?: string
  summary: string
  href?: string
  highlights?: readonly string[]
}

export type InterestGroup = {
  title: string
  description: string
}

export type Profile = {
  name: string
  statement: string
  introduction: string
  location: string
  navigation: readonly { label: string; href: `#${string}` }[]
  links: {
    resume: string
    linkedin: string
    github: string
    email: `mailto:${string}`
  }
  current: readonly InterestGroup[]
  experience: readonly TimelineEntry[]
  education: readonly TimelineEntry[]
  interests: readonly InterestGroup[]
}

export const profile = {
  name: 'Tony Wang',
  statement: 'I explore mathematics, computation, and the systems they help us understand.',
  introduction: 'I am a data scientist working in quantitative finance, currently based in London.',
  location: 'London, United Kingdom',
  navigation: [
    { label: 'About', href: '#about' },
    { label: 'Writing', href: '#writing' },
    { label: 'Experience', href: '#experience' },
    { label: 'Education', href: '#education' },
    { label: 'Contact', href: '#contact' },
  ],
  links: {
    resume: '/assets/resume/resume.pdf',
    linkedin: 'https://www.linkedin.com/in/twan3617/',
    github: 'https://github.com/twan3617',
    email: 'mailto:tonywang205@yahoo.com.au',
  },
  current: [
    {
      title: 'Work',
      description: 'Data science in quantitative finance at Coolabah Capital Investments.',
    },
    {
      title: 'Place',
      description: 'Living and working in London after a year in Coolabah’s Sydney office.',
    },
    {
      title: 'Exploring',
      description: 'Mathematics, computation, statistics, and the ideas connecting them.',
    },
  ],
  experience: [
    {
      period: 'May 2024 — Present',
      title: 'Data Scientist',
      subtitle: 'Coolabah Capital Investments',
      location: 'London, United Kingdom',
      summary: 'Coolabah is a fixed-income asset manager. I joined the Data Science team in Sydney, spent my first year there, and am now one of two data scientists in London.',
    },
    {
      period: 'February 2023 — April 2024',
      title: 'Data Analyst',
      subtitle: 'Quantium',
      summary: 'I joined through the graduate program and later became a Contributor Data Analyst, working across product analytics and consulting before moving into Quantium’s Health vertical to support UK health and pharmaceutical business development.',
      highlights: ['SQL', 'Snowflake', 'Spark', 'Python', 'Winner, Quantium 2023 GenAI Hackathon'],
    },
    {
      period: 'August 2021 — March 2022',
      title: 'Machine Learning Research Assistant',
      subtitle: 'University of Sydney',
      summary: 'I helped develop a streaming matrix-profile proof of concept for detecting regime changes in noisy sensor data.',
    },
    {
      period: '2020 — 2022',
      title: 'Mathematics educator',
      subtitle: 'University of Sydney and QED Education',
      summary: 'I taught high-school and university mathematics, an unusually fulfilling chapter of my working life.',
    },
  ],
  education: [
    {
      period: '2021 — 2022',
      title: 'Master of Mathematics with Excellence',
      subtitle: 'UNSW',
      summary: 'Research on model-selection consistency for the Lasso and a root-log regulariser.',
      href: '/assets/thesis/masters_thesis_tw.pdf',
    },
    {
      period: '2018',
      title: 'Academic exchange',
      subtitle: 'University of California, Berkeley',
      summary: 'Studied partial differential equations, probability theory, real analysis, and discrete mathematics.',
    },
    {
      period: '2016 — 2020',
      title: 'BSc (Advanced), Honours in Pure Mathematics',
      subtitle: 'University of Sydney',
      summary: 'Research on pseudomonotone operators and anisotropic elliptic equations.',
      href: '/assets/thesis/honours_thesis_tw.pdf',
    },
  ],
  interests: [
    { title: 'Books', description: 'Crime, history, and culture.' },
    { title: 'Travel', description: 'Exploring London and Europe’s rich history.' },
    { title: 'Piano', description: 'Playing V.K’s “Pure White,” DJ Okawari’s “Flower Dance,” and “Melody of the Night.”' },
  ],
} as const satisfies Profile
