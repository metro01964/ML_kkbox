"""`import src` 必須讓 stdout／stderr 變成 UTF-8。

為什麼需要這個測試：`src/__init__.py` 裡那段 reconfigure 是 **import 副作用**，
看起來像可以安全刪掉的雜訊。刪掉之後 CI（ubuntu-latest，UTF-8 預設）依然全綠，
問題只會在讀者自己的繁體中文 Windows 上出現 —— 而且症狀極度誤導：artifact 已經
寫好、round-trip 也驗證過，卻在印最後一行成功訊息時拋 UnicodeEncodeError，
`make artifact-t7` 因此回報 exit 1。**一個成功的匯出看起來像失敗。**

所以這裡不測「編碼屬性等於 utf-8」——那在 Linux 上本來就成立，測不出東西。
改成開 subprocess 並**把 stdout 接成管線**，複製出當初真正的失敗條件：
stdout 不是終端機時，Python 會退回 locale 編碼（cp950），而 cp950 編不出 ✅。
"""

import subprocess
import sys

# scripts/ 底下實際用到的狀態字元，數千處。
MARKERS = "✅ ⚠️ ❌ 🚧"


def _run_piped(code: str) -> subprocess.CompletedProcess:
    """在子行程執行 code，stdout／stderr 都接成管線（不是終端機）。

    `capture_output=True` 就是關鍵 —— 它讓子行程的 stdout 不是 tty，
    Python 因此改用 locale 編碼。這正是 `make ... > log.txt`、CI 擷取輸出、
    以及本專案背景執行時的實際情境。
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_importing_src_makes_piped_stdout_survive_status_markers():
    r = _run_piped(f"import src; print('{MARKERS}')")
    assert r.returncode == 0, f"import src 後仍然編不出狀態字元：\n{r.stderr}"
    assert MARKERS in r.stdout


def test_importing_src_also_fixes_stderr():
    """錯誤路徑同樣會印 ❌ —— 修了 stdout 卻漏掉 stderr，等於沒修一半。"""
    r = _run_piped(f"import sys, src; print('{MARKERS}', file=sys.stderr)")
    assert r.returncode == 0, f"stderr 仍然編不出狀態字元：\n{r.stderr}"
    assert MARKERS in r.stderr


def test_the_failure_mode_is_real_on_this_machine():
    """沒有 `import src` 時到底會不會壞，取決於這台機器的 locale。

    在 UTF-8 的系統上（CI、macOS、多數 Linux）本來就不會壞，所以這個測試
    **不斷言它必須失敗** —— 那會讓 CI 紅燈。它只在真的壞掉時，確認壞的原因
    正是編碼，而不是別的問題被我們誤診。
    """
    r = _run_piped(f"print('{MARKERS}')")
    if r.returncode != 0:
        assert "codec can't encode" in r.stderr, f"預期是編碼問題，實際錯誤不同：\n{r.stderr}"
