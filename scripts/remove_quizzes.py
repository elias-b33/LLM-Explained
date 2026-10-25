"""One-time: strip quizzes from all module files and swap in the Placard."""
import glob
import os
import re

MOD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "modules")

for path in sorted(glob.glob(os.path.join(MOD_DIR, "m*.tsx"))):
    s = open(path, encoding="utf-8").read()
    orig = s

    # Quiz import -> Placard + TAKEAWAYS imports
    s = s.replace(
        "import { Quiz } from '@/primitives/Quiz'",
        "import { Placard } from '@/primitives/Placard'\nimport { TAKEAWAYS } from '@/data/takeaways'",
    )
    # Drop the QuizSpec type import
    s = re.sub(r"import type \{ QuizSpec \} from '@/types'\n", "", s)
    # Remove the quiz const block (closing brace at column 0)
    s = re.sub(r"const quiz: QuizSpec = \{.*?\n\}\n", "", s, flags=re.DOTALL)
    # Swap the <Quiz/> element for the <Placard/>, carrying the slug across
    s = re.sub(
        r"<Quiz spec=\{quiz\} moduleId=\"([^\"]+)\" />",
        r"<Placard takeaways={TAKEAWAYS['\1']} />",
        s,
    )
    # Tidy any tripled blank lines left behind
    s = re.sub(r"\n\n\n+", "\n\n", s)

    if s != orig:
        open(path, "w", encoding="utf-8").write(s)
        print("updated", os.path.basename(path))
    else:
        print("unchanged", os.path.basename(path))
