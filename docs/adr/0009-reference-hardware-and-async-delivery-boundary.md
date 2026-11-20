# Reference Hardware Tier and Asynchronous Commercial Delivery Boundary

## Context
Commercializing edge model compression at entry-level pricing (US$49–99) carries acute risk of scope creep when clients request physical sensor wiring, custom PCB driver debugging, or live video consulting calls across global timezones, jeopardizing academic commitments and unsustainable hourly wages.

## Decision
We enforce a strict boundary on commercial deployment engagements:
1. Deliverables are strictly bound to **Reference Hardware Tiers** (ESP32 DevKit v1, STM32 Nucleo-F401RE) running pre-recorded in-memory test vectors or static buffers. Custom PCB bring-up, sensor driver development, and physical wiring are explicitly out of scope.
2. Client communication is strictly **asynchronous** (GitHub Private Issues / Email) with a 24–48 business hour response SLA and 5–7 calendar day turnaround, eliminating live video calls.
3. Payment escrow is routed via established platforms (Fiverr / Wise) verified under legal guardian stewardship to ensure cross-border regulatory compliance.

## Consequences
- Eliminates hardware-in-the-loop debugging risks on unverified client hardware.
- Protects academic and rest schedules from cross-timezone interruption.
- Caps project turnaround to standardized, reproducible C99 header verification and numerical parity reporting.
