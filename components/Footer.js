
import React from 'react';
// import styles from './global.css';

const Footer = () => {
  return (
    <>
      <div className="container">
      <footer className="fixed bottom-0 left-0 z-20 w-full p-4 bg-white border-t border-gray-200 shadow md:flex md:items-center md:justify-between md:p-6 dark:bg-gray-800 dark:border-gray-600">
    <span className="text-sm text-gray-500 sm:text-center dark:text-gray-400">© 2024 <a href="/" className="hover:underline"> Mathematics, Statistics and Everything Else</a>
    </span>
    <ul className="flex flex-wrap items-center mt-3 text-sm font-medium text-gray-500 dark:text-gray-400 sm:mt-0">
        <li>
            <a href="/" className="hover:underline me-4 md:me-6">About Me</a>
        </li>
        <li>
            <a href="/assets/resume/resume.pdf" className="hover:underline me-4 md:me-6">Resume</a>
        </li>
        <li>
            <a href="https://www.linkedin.com/in/twan3617/" className="hover:underline me-4 md:me-6">LinkedIn</a>
        </li>
        <li>
            <a href="https://github.com/twan3617" className="hover:underline me-4 md:me-6">GitHub</a>
        </li>
        <li>
            <a href="mailto: tonywang205@yahoo.com.au" className="hover:underline">Email</a>
        </li>
    </ul>
    </footer>      
    </div>
    </>
  );
};

export default Footer;