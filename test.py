import requests
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

URL = "http://100.94.254.20:8000/scanned"
PDF_FILE = "Scanned_Answersheet.pdf"
MODEL_ANSWER_ID = "6ab2547f4082b27c89cf4982"

TOTAL_REQUESTS = 25
OUTPUT_FILE = "responses.txt"

barrier = Barrier(TOTAL_REQUESTS)


def send_request(request_number):
    # Wait until all requests are ready
    barrier.wait()

    with open(PDF_FILE, "rb") as f:
        response = requests.post(
            URL,
            files={
                "myFile": (
                    PDF_FILE,
                    f,
                    "application/pdf"
                )
            },
            data={
                "preprocess": "true",
                "model_answer_id": MODEL_ANSWER_ID
            },
            headers={
                "accept": "*/*"
            },
            timeout=None
        )

    try:
        result = response.json()
        response_text = json.dumps(result, indent=2, ensure_ascii=False)
    except Exception:
        response_text = response.text

    return request_number, response_text


if __name__ == "__main__":

    print(f"Sending {TOTAL_REQUESTS} requests simultaneously...")

    with ThreadPoolExecutor(max_workers=TOTAL_REQUESTS) as executor:
        futures = [
            executor.submit(send_request, i)
            for i in range(1, TOTAL_REQUESTS + 1)
        ]

        results = [future.result() for future in futures]

    # Keep request order
    results.sort(key=lambda x: x[0])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for request_number, response_text in results:
            f.write(f"Request {request_number}\n")
            f.write(response_text)
            f.write("\n")
            f.write("=" * 80)
            f.write("\n")

    print(f"Done. Complete responses saved to: {OUTPUT_FILE}")