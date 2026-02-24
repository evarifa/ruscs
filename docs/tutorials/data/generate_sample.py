"""
Generate a synthetic sample of sample-activity_time.txt with the structure:
    userA  userB  timestamp  interaction
where interaction is one of: MT (mention), RT (retweet), RE (reply).

The sample mimics a real dataset's 8-day temporal structure (2012-07-01 to 2012-07-08)
with realistic bursts of new-user activity around day 4,
but uses entirely synthetic user IDs and timestamps.

Run:  python generate_sample.py
Output: sample-activity_time.txt  (same directory)
"""

import numpy as np
import os

rng = np.random.default_rng(42)

# --- Parameters ---
N_USERS   = 5_000          # synthetic unique users
N_EVENTS  = 80_000         # total interaction events
START_TS  = 1_341_100_800  # 2012-07-01 00:00:00 UTC
DURATION  = 8 * 24 * 3600  # 8 days in seconds
INTERACTIONS = ["MT", "RT", "RE"]

# --- Simulate user "join" times ---
# Most users join uniformly; a burst of new users appears on day 4 (the announcement)
day4_start = START_TS + 3 * 24 * 3600
day4_end   = START_TS + 4 * 24 * 3600

n_burst   = int(N_USERS * 0.35)   # 35% join during the day-4 burst
n_uniform = N_USERS - n_burst

uniform_join = rng.uniform(START_TS, START_TS + DURATION, n_uniform)
burst_join   = rng.uniform(day4_start, day4_end, n_burst)
all_join = np.concatenate([uniform_join, burst_join])
rng.shuffle(all_join)

user_ids = rng.choice(range(100_000, 999_999), size=N_USERS, replace=False)
user_join = dict(zip(user_ids, all_join))

# --- Generate interaction events ---
# Each user generates interactions starting from their join time
rows = []
users = list(user_ids)

for _ in range(N_EVENTS):
    # Pick a random user A
    ua = rng.choice(users)
    t_join = user_join[ua]

    # Activity decays exponentially after joining; higher during burst window
    lifetime = rng.exponential(scale=2 * 24 * 3600)  # avg 2-day activity window
    t_event  = t_join + rng.uniform(0, max(lifetime, 3600))

    # Clamp within dataset window
    t_event = min(t_event, START_TS + DURATION - 1)

    # Pick a random user B (different from A)
    ub = ua
    while ub == ua:
        ub = rng.choice(users)

    interaction = rng.choice(INTERACTIONS)
    rows.append((ua, ub, int(t_event), interaction))

# Sort by timestamp
rows.sort(key=lambda x: x[2])

# --- Write output ---
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sample-activity_time.txt")
with open(out_path, "w") as f:
    for ua, ub, ts, itype in rows:
        f.write(f"{ua} {ub} {ts} {itype}\n")

print(f"Written {len(rows)} events to {out_path}")
