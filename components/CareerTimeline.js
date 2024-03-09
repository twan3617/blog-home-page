const CareerTimeline = () => {
    return       <section className="career p-5">
    <h2 className="text-3xl font-mono text-center text-dark-teal  p-5">
        Career
    </h2>
    <ol className="relative border-l border-black">
        <li className="mb-10 ml-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -left-1.5 border border-white bg-black"></div>
            <time className="font-mono mb-1 text-sm  leading-none ">February 2024 - Present</time>
            <h3 className="font-mono  text-lg font-bold">Contributor Data Analyst @ Quantium</h3>
            <p className="indent-5">
                I completed my graduate program and got promoted! I am now working in the Health vertical at Quantium, 
                with a particular focus on supporting business development opportunities in the UK health and pharmaceutical industries. 
            </p>
        </li>
        <li className="mb-10 ml-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -left-1.5 border border-white bg-black"></div>
            <time className="font-mono mb-1 text-sm  leading-none ">February 2023 - February 2024</time>
            <h3 className="font-mono  text-lg font-bold">Graduate Data Analyst @ Quantium</h3>
            <p className="indent-5">
                My time at Quantium exposed me to tons of cool technologies, like&nbsp;
                <span className="text-teal">SQL</span>,&nbsp;
                 <span className="text-teal">Snowflake</span>,&nbsp;
                 <span className="text-teal">Spark</span>,&nbsp; 
                 <span className="text-teal">Python</span> and&nbsp;
                 <span className="text-teal">Excel</span>. I was also a part of the winning team in <span className="text-teal">Quantium's 2023 GenAI Hackathon!</span>
            </p>
            <p>
              I worked mainly in Product Analytics for Q.Refinery (bank transaction labeller and customer attribution), Q.Checkout (Snowflake costing and user analysis), and in consulting for global insurance companies. 
            </p>
        </li>
        <li className="mb-10 ml-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -left-1.5 border border-white bg-black"></div>
            <time className="font-mono mb-1 text-sm  leading-none ">August 2021 - March 2022</time>
            <h3 className="font-mono  text-lg font-bold">ML Research Assistant @ USYD</h3>
            <p className="mb-4 text-base indent-5">
                Using <a href="https://www.cs.ucr.edu/~eamonn/MatrixProfile.html" className="underline">matrix profiling</a>&nbsp;techniques, we developed a streaming data proof-of-concept to identify regime changes in noisy sensor data, with real world applications in 
                detecting major structural damage automatically as they occur.
                See our&nbsp;
                <a href="https://sjmluo.github.io/anomaly/" className="underline">website</a> and&nbsp;
                <a href="https://github.com/sjmluo/Contextually_Situated_Anomaly_Detection" className="underline">repository</a>!
            </p>
        </li>
        <li className="mb-10 ml-4">
            <div className="absolute w-3 h-3 bg-gray-200 rounded-full mt-1.5 -left-1.5 border border-white bg-black"></div>
            <time className="font-mono mb-1 text-sm  leading-none ">2020 - 2022</time>
            <h3 className="font-mono  text-lg font-bold">Casual Academic @ USYD</h3>
            <h3 className="font-mono  text-lg font-bold">Mathematics Teacher @ QED Education</h3>
            <p className="mb-4 text-base indent-5">
                I taught high-school and university-level mathematics for a few years. 
                Although I did not make a career of it, education was a very fulfilling part of my life.
            </p>
        </li>
    </ol>
</section>
}

export default CareerTimeline