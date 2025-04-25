'''
(Fine-tuning) retrieved documents를 embedding할 수 있는 형태로 변환
'''
import json
import argparse

def convert_format(line_data):
    return {
        "question": (
            line_data["question"]
        ),
        "answer": line_data["answer"],
        "retrieved_docs": line_data["retrieved_docs"]
    }

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-if", type=str, required=True)
    parser.add_argument("--output_file", "-of", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_argument()

    converted_data = []
    with open(args.input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                converted = convert_format(item)
                converted_data.append(converted)
            except Exception as e:
                print(f"[에러] 변환 실패: {e}\n→ 라인 내용: {line[:100]}...")

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(converted_data, outfile, ensure_ascii=False, indent=2)

    print(f"[완료] {len(converted_data)}개의 항목을 리스트 형태로 저장했습니다!: {args.output_file}")
