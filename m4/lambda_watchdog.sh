#!/usr/bin/env bash
# Terminate every running Lambda instance after N hours. Runs on the Mac, not the box.
# Lambda has no spend limit (deeptalk.lambda.ai/t/4121), and billing stops only at
# terminate -- a shut-down box still bills -- so this is the cap.
#
#   LAMBDA_API_KEY=... m4/lambda_watchdog.sh 3      # terminate everything in 3 h
#   LAMBDA_API_KEY=... m4/lambda_watchdog.sh 0      # terminate everything now
#
# Run it in its own terminal the moment the box is launched; closing that
# terminal kills the watchdog, so leave it open.
set -euo pipefail
hours=${1:?usage: lambda_watchdog.sh HOURS}
: "${LAMBDA_API_KEY:?set LAMBDA_API_KEY (Lambda console -> API keys)}"
api=https://cloud.lambda.ai/api/v1
auth=(-H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json")

ids() { curl -sf "${auth[@]}" "$api/instances" | python3 -c '
import json,sys; print(json.dumps([i["id"] for i in json.load(sys.stdin)["data"]]))'; }

echo "watchdog: $(ids) running; terminating all in $hours h at $(date -v+"${hours}"H '+%H:%M')"
sleep "$((hours * 3600))"
live=$(ids)
if [ "$live" = "[]" ]; then echo "watchdog: nothing running"; exit 0; fi
curl -sf "${auth[@]}" -X POST "$api/instance-operations/terminate" \
  -d "{\"instance_ids\": $live}" >/dev/null
echo "watchdog: terminated $live at $(date '+%H:%M')"
