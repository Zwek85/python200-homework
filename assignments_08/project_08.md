# My project in supabase.com is set up as instructed.

# AWS Pricing Calculator

## Cost estimates (US East, N. Virginia, on-demand)

**Scenario A (lightweight compute):** A `t3.micro` running about 160 hours per month costs roughly **$1.66/month** (about $0.0104/hour).

**Scenario B (heavy analytics):** About **$2,380/month** in total:
- `p3.2xlarge` running 730 hours at $3.06/hour: about **$2,234**
- RDS `db.m5.large` running 730 hours at about $0.171/hour: about **$125**
- S3 Standard, 1 TB at about $0.023/GB: about **$24**

The numbers surprised me in two ways. The small instance is nearly free, but the GPU instance alone costs over 90% of Scenario B's total. The database and the storage, which feel like the "serious" parts of the setup, were small by comparison.

## How the scenarios compare
Scenario B costs about 1,400 times more than Scenario A, and nearly all of that gap comes from the GPU. A GPU instance is only worth it when the work really needs parallel GPU compute (like deep learning training), and even then it makes sense to run it only for the hours you need and shut it down afterward. Running one 24/7 for a workload a CPU could handle is an expensive mistake.

#video: https://www.youtube.com/watch?v=gz4KUMookvo&t=22s