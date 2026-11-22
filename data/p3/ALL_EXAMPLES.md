# Six original e.text passages, paired with e.tab CSV paths

The wording below is copied from the user-supplied originals, not generated descriptions. PDF hard line breaks are maintained.

## E1 — Fuel selector / imbalance / engine starvation
**Domain:** aviation  
**Source:** https://www.atsb.gov.au/investigations/ao-2021-053  
**Location:** ATSB Safety analysis → Fuel imbalance, PDF page 18 (printed page 14)  
**e.tab:** `examples/E1.e_tab.csv` (12 source-backed rows)

```text
Analysis of CCTV imagery indicated it was likely that VH-WMM had a substantial quantity of fuel 
in the right wing, and that the pilot would have been aware of the imbalance after physically using 
a dipstick to check the aircraft fuel tanks during the pre-flight checks on the ground outside the 
hangar. 
The fuel selector was found to be on the left tank after the accident and there was no video 
evidence of the pilot changing it during the flight. Therefore, the fuel selector was likely on the left 
tank during take-off. Due to the fuel imbalance, the left tank likely had minimal fuel to sustain 
engine operation during the climb out, which would result in the engine being starved of fuel, lose 
power, and begin to surge, and eventually stop. This is consistent with no fuel being found in the 
left tank after the accident.
```

## E2 — Inadequate glide and water impact
**Domain:** aviation  
**Source:** https://www.atsb.gov.au/investigations/ao-2021-053  
**Location:** ATSB Safety analysis → Collision with water, PDF page 19 (printed page 15)  
**e.tab:** `examples/E2.e_tab.csv` (14 source-backed rows)

```text
It is likely that the pilot never intended to ditch the aircraft. Rather, the reduced speed of the 
aircraft below the glide speed, exacerbated by the undercarriage extension, led to a reduced glide 
range and inability to reach the runway at Redcliffe. The sounding of the ‘stall’ alert just before the 
collision indicates the aircraft could no longer maintain lift and the aircraft collided with water 3 
seconds later.
```

## E3 — Gear extension / inversion / exit difficulty
**Domain:** aviation  
**Source:** https://www.atsb.gov.au/investigations/ao-2021-053  
**Location:** ATSB Findings → Contributing factors, PDF page 21 (printed page 17)  
**e.tab:** `examples/E3.e_tab.csv` (12 source-backed rows)

```text
• During the return to the airfield, the pilot extended the undercarriage, contributing to the aircraft 
inverting when it collided with water. This likely resulted in occupant disorientation, difficulty in 
operating the exits, and reduced their ability to escape.
```

## E4 — Underlying storage / Workers KV / dependent services
**Domain:** software  
**Source:** https://blog.cloudflare.com/cloudflare-service-outage-june-12-2025/  
**Location:** Cloudflare 2025-06-12, introduction, first two substantive paragraphs  
**e.tab:** `examples/E4.e_tab.csv` (16 source-backed rows)

```text
On June 12, 2025, Cloudflare suffered a significant service outage that affected a large set of our critical services, including Workers KV, WARP, Access, Gateway, Images, Stream, Workers AI, Turnstile and Challenges, AutoRAG, Zaraz, and parts of the Cloudflare Dashboard.

This outage lasted 2 hours and 28 minutes, and globally impacted all Cloudflare customers using the affected services. The cause of this outage was due to a failure in the underlying storage infrastructure used by our Workers KV service, which is a critical dependency for many Cloudflare products and relied upon for configuration, authentication and asset delivery across the affected services. Part of this infrastructure is backed by a third-party cloud provider, which experienced an outage today and directly impacted availability of our KV service.
```

## E5 — Failed remediation patch / renewed outage / rollback
**Domain:** software  
**Source:** https://blog.cloudflare.com/deep-dive-into-cloudflares-sept-12-dashboard-and-api-outage/  
**Location:** Cloudflare 2025-09-12 → How we responded, second paragraph  
**e.tab:** `examples/E5.e_tab.csv` (5 source-backed rows)

```text
After the Tenant Service began reporting healthy again and the API largely recovered, we still observed a considerable number of errors being reported from the service. We theorized that these were responsible for the ongoing Dashboard availability issues and made a patch to the service with the expectation that it would improve the API health and restore the dashboard to a healthy state. Ultimately this change degraded service further and was quickly reverted.
```

## E6 — Dashboard render bug / Tenant overload / authorization outage
**Domain:** software  
**Source:** https://blog.cloudflare.com/deep-dive-into-cloudflares-sept-12-dashboard-and-api-outage/  
**Location:** Cloudflare 2025-09-12 → What Happened, first three paragraphs  
**e.tab:** `examples/E6.e_tab.csv` (5 source-backed rows)

```text
We had an outage in our Tenant Service API which led to a broad outage of many of our APIs and the Cloudflare Dashboard.

The incident’s impact stemmed from several issues, but the immediate trigger was a bug in the dashboard. This bug caused repeated, unnecessary calls to the Tenant Service API. The API calls were managed by a React useEffect hook, but we mistakenly included a problematic object in its dependency array. Because this object was recreated on every state or prop change, React treated it as “always new,” causing the useEffect to re-run each time. As a result, the API call executed many times during a single dashboard render instead of just once. This behavior coincided with a service update to the Tenant Service API, compounding instability and ultimately overwhelming the service, which then failed to recover.

When the Tenant Service became overloaded, it had an impact on other APIs and the dashboard because Tenant Service is part of our API request authorization logic.  Without Tenant Service, API request authorization can not be evaluated.  When authorization evaluation fails, API requests return 5xx status codes.
```
