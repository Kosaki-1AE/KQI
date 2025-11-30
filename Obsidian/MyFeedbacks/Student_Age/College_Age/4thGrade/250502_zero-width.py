url = "https://www.kanazawa-it.ac.jp/"
invisible = '\u200B'
modified_url = url.join(invisible)
message = f"""こんな感じ{modified_url}らしいんだけど、なんかわかったやついる？いねぇよなぁ？"""
with open("250502_検証結果.txt", "w", encoding="utf-8") as f:
    f.write(message)
    
def extract_invisible(message):
# ゼロ幅を取り除かず、不可視文字だけ拾って復元
    return ''.join(c for c in message if ord(c) in [0x200B, 0x200C, 0x200D])

# またはURL部分だけ抜き出す（可視化）
def reveal_url(message):
    return ''.join(c for c in message if c not in ('\u200B', '\u200C', '\u200D'))

recovered_invisible = extract_invisible(message) #これだと空白箇所だけを抜き取っとるんで分からんです。空白になるだけよ。
recovered_url = reveal_url(message)

answer_message = f"""\nこれね、正解はぁ...「こんな感じ」の部分に{url}が含まれてたんす....まさかURLが入ってるとは気づかんだろ(AI以外にはな....)"""

with open("250502_検証結果.txt", "a", encoding="utf-8") as f:
    f.write(answer_message)