# Cloud Concepts (short version)

# Q1: Core economic model**
Cloud turns infrastructure from a big upfront purchase (CapEx) into a pay-as-you-go rental (OpEx). You pay only for what you use instead of buying servers sized for peak demand and paying for them even when they sit idle.

# Q2: Vertical vs. horizontal scaling**
- **Vertical:** make one machine bigger (more CPU, RAM, GPU). Good for a database that needs more memory.
- **Horizontal:** add more machines and split the work. Good for a web API handling lots of traffic.

Scenarios:
- **Viral web app:** Horizontal, because traffic can be spread across many servers.
- **Model training needing a faster GPU and more RAM:** Vertical, because you need one more powerful machine.
- **10 to 10,000 files, splittable work:** Horizontal, because files can be processed in parallel on many machines.

# Q3: Classification**

| Item | Type | Why |
|---|---|---|
| Gmail | SaaS | Finished app you just use |
| Azure Virtual Machines | IaaS | You rent raw servers |
| AWS S3 | IaaS | Raw storage infrastructure |
| GitHub Codespaces | PaaS | Ready-made dev environment, you bring code |
| Snowflake | SaaS | Fully managed data platform you query with SQL |
| Supabase | BaaS | Ready-made backend (database, auth, storage) |

**Definitions**
- **IaaS:** You rent basic building blocks (servers, storage, networking). Example: Azure VMs. *You manage:* the OS, runtime, app, and data.
- **PaaS:** You get a managed platform to run your code. Example: GitHub Codespaces. *You manage:* your code and data.
- **SaaS:** You use a finished application. Example: Gmail. *You manage:* just your own content and settings.

**Main point:** going from IaaS to PaaS to SaaS, you manage less and the provider manages more, trading control for convenience.

# Q4: Managed data platforms
A managed data platform like Databricks or Snowflake is a ready-made service for storing, processing, and analyzing data. It handles the infrastructure underneath (servers, scaling, tuning, security) so you can focus on working with data. Using AWS or GCP directly means assembling those pieces yourself from lower-level services like storage, compute, and networking.

What you gain: Faster setup, less operational work, built-in scaling and optimization, and a single tool that already fits together.
What you give up: Some control and flexibility, plus a higher price for the convenience. You're also more tied to that vendor's tools and way of doing things (vendor lock-in).

# Q5: When cloud is probably not the right choice
When you have a large, steady, predictable workload that runs constantly. In that case, owning your own hardware can end up cheaper than renting indefinitely, since you aren't benefiting from the cloud's pay-for-what-you-use flexibility.

# Cloud Landscape 

# Q1: The three hyperscalers**
- **AWS (Amazon Web Services):** The largest and broadest catalog of services, with the most mature ecosystem. Used by startups, tech companies, and nearly everyone who wants maximum choice.
- **Microsoft Azure:** Tight integration with Microsoft products (Office 365, Active Directory, Windows Server). Used by large enterprises already invested in Microsoft.
- **Google Cloud Platform (GCP):** Strength in data analytics, machine learning, and Kubernetes. Used by data-heavy and AI-focused organizations.

# Q2: Why the course switched from Azure to Supabase**

1. **Complexity:** Azure has a steep learning curve and a lot of setup (accounts, subscriptions, resource groups) before you can do anything useful.
2. **Cost and access:** Azure can require billing setup and can cost money, while Supabase has a free tier that's easy for students to use.
3. **Fit for the course:** Supabase gives you a ready-to-use Postgres database and API, which matches what a data course actually needs.

**My reflection:** Don't pick a tool because it's the biggest or most famous. Evaluate it against your project: how fast can you get started, what does it cost, how much setup and maintenance does it demand, and does it do what you actually need? The simplest tool that fits is usually the best choice.

# Q3: Service categories**

| Scenario | Category | Example product |
|---|---|---|
| 10 TB of images retrieved by filename | Object storage | AWS S3 |
| 4-hour GPU training job, then shut down | GPU compute (on-demand VMs) | AWS EC2 GPU instances (or Google Colab) |
| Web API that scales up and down automatically | Serverless compute | AWS Lambda (or Google Cloud Run) |
| Send structured data to an LLM, get text back | LLM API | Anthropic Claude API (or OpenAI API) |

# Q4: My own project and stack**
**Project:** A weather data tracker that collects daily readings, stores them, and shows trends on a dashboard.
- **Storage:** Supabase (managed Postgres database)
- **Collection:** AWS Lambda (runs a small script daily on a schedule)
- **Dashboard:** Streamlit Community Cloud (hosts the app)

**Benefit of consolidating to one provider:** Simpler billing, one login, easier integration between services, and one support contact. Services from the same provider usually work together smoothly and can avoid data transfer fees between them.

**What you'd give up:** The freedom to choose the best tool for each job, plus some cost savings, since a single provider's version of a service may be pricier or weaker. You'd also be more locked in, making it harder to switch later.