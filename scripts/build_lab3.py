"""Build the laboratory 3 notebook with concise calls to source modules.

    python scripts/build_lab3.py              # new notebook, outputs cleared
    python scripts/build_lab3.py --text-only  # replace the text, keep outputs

A text-only update is allowed only when the code cells are unchanged, and
it ends with the statements check: the new text must agree with the stored
results.
"""

import argparse
from pathlib import Path

import nbformat

from emlab.lab3_config import (
    DEFAULT_CONFIG,
    KERNEL_DISPLAY_NAME,
    KERNEL_NAME,
    NOTEBOOK_PATH,
)

ROOT = Path(__file__).resolve().parents[1]
CELLS = (
    (
        "markdown",
        r"""# Лабораторная №3. Метод Метрополиса — Хастингса для графов

Продолжение [ЛР2](lab2_interventions.ipynb): те же модели, данные,
критерий удаления и интервенция. Во второй работе каждое слагаемое
проверялось отдельно в одной фиксированной модели. Здесь графы модели
**семплируются**, а интервенция входит и в целевую функцию,
и в предложение следующего графа.

| Пункт задания | Выбор |
|---|---|
| 1. Уравнение ЛР1 | Два связанных контура; дисперсия кварцевого стекла и воды |
| 2. Критерий ЛР2 | Удалить элемент, если после переподгонки $\Delta RMSE\le\tau$ и не более 5% бутстреп-оценок лежат на границах; бутстреп, шум $\alpha s(x)\xi$, расстояние $W_1$ |
| 3. Алгоритм | Метрополис — Хастингс по графам |

## 1. Граф, параметры, данные

Модель $y=f_G(w,x)$, наблюдений $m$. Граф $G$ — набор рёбер из $K$
кандидатов, $|G|$ — число рёбер. На ребре $e$ один параметр $w_e$;
если ребра нет, параметр закреплён на значении удаления $w_e^0$ — том
же, что при проверке удаления в ЛР2 (ноль; для $R_2$ — $10^{-6}$ Ом).
Параметры модели — $w$, структурный параметр — $G$, гиперпараметры —
масштаб ошибки $s(x)$ из ЛР2, уровень интервенции $\alpha$ и вес
неустойчивости $\lambda$.

**Контуры.** $x$ — время, $y$ — ток вторичного контура $i_2=\dot q_2$.
Уравнение ЛР1 для зарядов $q=(q_1,q_2)$ со всеми видами связи контуров:

$$
\begin{bmatrix}L_1&M\\M&L_2\end{bmatrix}\ddot q
+\begin{bmatrix}R_1&R_m\\R_m&R_2\end{bmatrix}\dot q
+\begin{bmatrix}1/C_1&S_m\\S_m&1/C_2\end{bmatrix}q
=\begin{bmatrix}1\\E_2\end{bmatrix}\cos\omega x .
$$

В ЛР2 оценивались $M$ и $R_2$. Добавлены три ребра, которых в данных
нет (тот же приём, что $c_0$ в ЛР2): резистивная связь $R_m$, ёмкостная
$S_m$ и наводка источника во вторичном контуре $E_2$.

**Дисперсия.** $x$ — длина волны, $y=n^2-1$, $n$ — показатель
преломления. К резонансам Зельмейера $\varphi_j(x)=x^2/(x^2-b_j)$
добавлены полиномиальные члены:
$y=\sum_j a_j\varphi_j(x)+\sum_{p\in\{0,2,-2,-4\}}c_px^p$.
Данные — опубликованные формулы дисперсии с добавленной ошибкой,
как в ЛР2.""",
    ),
    (
        "code",
        """from pathlib import Path

from emlab import lab3_experiment, lab3_graphs, lab3_plots, lab3_reporting
from emlab.lab3_config import DEFAULT_CONFIG as CONFIG

ROOT = Path.cwd()
ROOT = ROOT.parent if ROOT.name == "notebooks" else ROOT
lab3_reporting.show_environment(CONFIG)""",
    ),
    (
        "code",
        """problems = lab3_graphs.prepare_problems(ROOT)
lab3_plots.plot_candidates(ROOT, problems)
lab3_reporting.show_edges(problems)""",
    ),
    (
        "markdown",
        r"""Сплошные рёбра порождали данные, пунктирные — нет. Связи контуров
взаимные, поэтому их граф не ациклический: ациклической была бы его
развёртка по времени. Графы дисперсии ациклические.

## 2. Целевая функция

**Первый уровень, граф фиксирован.** Параметры имеющихся рёбер
оцениваются взвешенным методом наименьших квадратов с границами
на параметры:
$\chi^2(G)=\min_w\bigl\|(y-f_G(w,x))/s(x)\bigr\|^2$,
$\hat w$ — точка минимума.

**Второй уровень, выбор графа.** Графы семплируются из распределения
$\pi(G)\propto\exp(-J(G))$,

$$
J(G)=\underbrace{\tfrac12\chi^2(G)}_{\text{невязка}}
+\tfrac12\ln m\,\Bigl(\underbrace{|G|}_{\text{сложность}}
+\lambda\underbrace{\sum_{e\in G}u_e(G)}_{\text{неустойчивость к интервенции}}\Bigr).
$$

Первые два слагаемых — приближение Шварца для $-\ln p(y\mid G)$
с точностью до константы и $O(1)$ при известном $s(x)$ и параметрах
внутри границ. Третье — регуляризатор по интервенции,
$u_e\in[0,1]$ — неустойчивость ребра. Он вычисляется по тем же данным,
поэтому $\pi$ — распределение Гиббса, а не апостериорное $p(G\mid y)$.

**Устойчивость ребра.** Интервенция — шум в отклике:
$y'=y+\alpha s(x)\xi$, $\xi\sim N(0,1)$. $P_e$ и $P'_e$ —
бутстреп-распределения $w_e$ до и после интервенции (обе процедуры
из ЛР2), $\delta_{w_e^0}$ — распределение, сосредоточенное в значении
удаления:

$$
\rho_e=\frac{\overline{W_1(P_e,P'_e)}}{W_1(P_e,\delta_{w_e^0})},
\qquad u_e=\min(1,\rho_e),
$$

черта — среднее по @REPEATS@ реализациям $\xi$. Числитель — расстояние
из ЛР2, сдвиг параметра при интервенции. Знаменатель равен среднему
$|w_e-w_e^0|$ по $P_e$ — это сдвиг при удалении ребра.
Порог $\rho_e=1$: интервенция сдвигает параметр не меньше, чем удаление
ребра. Если параметр не определён данными (не меняется при бутстрепе
или более 5% оценок лежат на границе до или после интервенции — порог
ЛР2), то $u_e=1$.

В модели кварцевого стекла с $c_0$ нормировка ЛР2 на стандартное отклонение
$\sigma_e$ распределения $P_e$ не отделяет лишний параметр
от порождающих: при $\alpha=2$ отношение $W_1(P_e,P'_e)/\sigma_e$
равно 1.7–2.5 для всех четырёх параметров.

**Порог отклонения ребра.** $\Delta J_e$ — изменение $J$ при смене
состояния ребра $e$: удалении, если оно есть, и добавлении, если нет.
При удалении
$\Delta J_e\approx\tfrac12\bigl(\Delta\chi^2_e-(1+\lambda u_e)\ln m\bigr)$,
где $\Delta\chi^2_e$ — изменение $\chi^2$ (изменение $u$ остальных рёбер
не учтено). Удаление выгодно, если
$\Delta\chi^2_e\le(1+\lambda u_e)\ln m$: без интервенции порог равен
$\ln m$, для неустойчивого ребра он выше в $1+\lambda$ раз. Правило ЛР2
решает тот же вопрос по $\Delta RMSE$ на отложенной сетке, один раз
и детерминированно.

**Что измеряет критерий.** Для хорошо определённого параметра
$\rho_e\approx\kappa/|z_e|$, где $z_e=(\hat w_e-w_e^0)/\sigma_e$ —
статистика Вальда, а $\kappa=W_1(P_e,P'_e)/\sigma_e$ зависит от $\alpha$
и реализации шума. Шум в отклике показывает, насколько прочно ребро
определено данными, а не направление связи, и в линейном приближении
не даёт информации сверх $\Delta\chi^2_e$: регуляризатор усиливает
штраф за слабо определённые рёбра. Правила do-исчисления — условия
на граф; вопрос правила 3 (можно ли игнорировать интервенцию) здесь
решается только эмпирически, по сдвигу оценки.

## 3. Алгоритм

Источник: Метрополис и др. (1953), Хастингс (1970); для графов —
structure MCMC: добавление и удаление ребра (Madigan, York, 1995),
обращение ребра (Giudici, Castelo, 2003).

| | Structure MCMC | В работе |
|---|---|---|
| Состояние | ациклический граф | набор рёбер из $K$ кандидатов |
| Ходы | добавить, удалить, обратить ребро | добавить, удалить, заменить ребро другим |
| Цель | $p(G\mid y)$ | $\exp(-J(G))$: невязка, сложность, неустойчивость |
| Предложение | равномерное | удаление — по интервенции на текущем графе |
| Результат | вероятности рёбер | сохранённые графы цепей и $G^*=\arg\min J$ по ним |

1. $G_0$ — полный, пустой или случайный граф; $t$ — номер шага.
2. Для $G_t$: подгонка $\hat w$, бутстреп, интервенция, бутстреп;
   получаем $u_e(G_t)$ и $J(G_t)$.
3. Предложить $G'$ из распределения $Q(G'\mid G_t)$. С вероятностью
   $1-p_s$ сменить состояние одного ребра, выбранного с весом
   $v_e=\varepsilon+u_e(G_t)$, если ребро есть, и $v_e=1$, если его нет:
   неустойчивые рёбра чаще предлагаются к удалению. С вероятностью $p_s$
   заменить случайное имеющееся ребро случайным отсутствующим.
4. Повторить шаг 2 для $G'$.
5. Принять $G_{t+1}=G'$ с вероятностью
   $A(G_t,G')=\min\Bigl\{1,\;\exp\bigl(J(G_t)-J(G')\bigr)\,\dfrac{Q(G_t\mid G')}{Q(G'\mid G_t)}\Bigr\}$,
   иначе $G_{t+1}=G_t$.
6. Сохранить $G_{t+1}$ и перейти к шагу 2.

Два этапа. **Без интервенции**: $\lambda=0$, $v_e\equiv1$.
**С интервенцией**: $\lambda=@WEIGHT@$, $\varepsilon=@FLOOR@$,
$\alpha=@ALPHA@$. На обоих $p_s=@SWAP@$.
$G^*=\arg\min J$ берётся по всем сохранённым графам, частоты —
без первых @BURN@ шагов (прогрев).

**Корректность.** При любом $G_0$ распределение $G_t$ сходится к $\pi$,
а частоты графов в цепи — к $\pi(G)$.

- *Обратимость.*
  $\pi(G)Q(G'\mid G)A(G,G')=\min\{\pi(G)Q(G'\mid G),\,\pi(G')Q(G\mid G')\}$
  симметрично по $G,G'$, значит $\pi$ стационарно. Смена одного ребра
  меняет $|G|$, замена — нет, поэтому каждый сосед получается ровно
  одним ходом и $Q$ — полная вероятность предложения.
- *Неприводимость.* $v_e\ge\varepsilon>0$ и $J$ конечна, поэтому
  любая смена одного ребра возможна; между любыми двумя графами
  не более $K$ таких ходов.
- *Апериодичность.* В пустом и полном графе замена невозможна, и цепь
  остаётся на месте с вероятностью $p_s>0$.

Конечная неприводимая апериодическая цепь сходится к единственному
стационарному распределению. Условие: $J$ и $v_e$ — функции только
графа. Поэтому индексы бутстрепа и шум интервенции $\xi$ общие для всех
графов, а результаты кешируются.

Реализация проверена тестами на случайно заданном распределении
по 16 графам с неравномерными весами: переходы цепи совпадают с точной
матрицей переходов, частоты сходятся к $\pi$.

## 4. Эксперимент

На каждом этапе три цепи по @STEPS@ шагов. Полным перебором $\pi$
вычислено без интервенции для всех задач, с интервенцией — для контуров
(32 графа). Все графы цепей сохраняются в `results/lab3/chains.npz`.""",
    ),
    (
        "code",
        """results = {
    name: lab3_experiment.run_problem(problem, CONFIG)
    for name, problem in problems.items()
}
summary = lab3_experiment.save_results(ROOT, results)""",
    ),
    (
        "markdown",
        r"""### 4.1. Правило ЛР2 на полном графе

Применяем правило удаления ЛР2 к графу со всеми пятью рёбрами.""",
    ),
    (
        "code",
        """rlc = results["rlc"]
lab3_reporting.show_lab2_rule(rlc, (True,) * len(rlc.problem.edges))""",
    ),
    (
        "markdown",
        r"""По $\Delta RMSE\le\tau$ удалимы $M$, $R_m$, $S_m$ и $E_2$: каждое
по отдельности заменимо остальными, включая порождающее $M$ — его
компенсирует $S_m$. Проверка границ при этом запрещает все удаления.
Проверка рёбер по одному в полном графе не приводит к $\{M,R_2\}$,
поэтому перебираем графы.

### 4.2. Устойчивость рёбер к интервенции""",
    ),
    (
        "code",
        """lab3_plots.plot_stability(
    ROOT, lab3_experiment.stability_cases(results), CONFIG.stability_levels
)""",
    ),
    (
        "markdown",
        r"""Графы — порождающие рёбра плюс одно лишнее. Порождающие рёбра
(сплошные линии) остаются более чем на порядок ниже порога при всех
показанных $\alpha$. Лишнее ребро (пунктир) пересекает порог при $\alpha$ между
1 и 2 для контуров и между 0.5 и 1 для кварцевого стекла. Далее $\alpha=2$,
как в примере ЛР2. Пересечение зависит от реализации шума: у $c_2$
в обеих задачах дисперсии $\rho_e$ чуть ниже 1 (таблица в 4.5),
штраф $u_e$ при этом почти полный.

### 4.3. Последовательность семплированных графов

С интервенцией, старт из полного графа. Показаны принятые графы
и $J(G_t)-J(G^*)$; между показанными шагами все предложения
отклонены. Узлы слева направо: источник, первичный контур,
вторичный контур.""",
    ),
    (
        "code",
        """lab3_plots.plot_sequence(
    ROOT,
    rlc,
    lab3_experiment.INTERVENTIONAL,
    "полный граф",
    CONFIG.sequence_steps,
)""",
    ),
    (
        "markdown",
        r"""Цепь сначала попадает в граф без $M$: ёмкостная связь с поправками
$R_m$ и $E_2$ описывает ток почти так же. Выходит она оттуда заменой
ребра на $M$, после чего лишние рёбра удаляются. Дальше цепь остаётся
в $\{M,R_2\}$ и изредка принимает одно лишнее ребро.

### 4.4. Сходимость и проверка корректности""",
    ),
    (
        "code",
        """lab3_plots.plot_convergence(ROOT, rlc)
lab3_reporting.show_proposals(rlc)""",
    ),
    (
        "markdown",
        r"""Слева — целевая функция в начале цепи: все три старта приходят
к одному минимуму. Справа — расстояние полной вариации (TV) между
частотами графов в цепи и точным $\pi$. Без интервенции оно убывает
примерно как $t^{-1/2}$ (серая линия). С интервенцией почти вся масса
$\pi$ лежит в одном графе,
поэтому малое расстояние не говорит о точности частот редких графов;
в 4.5 частоты рёбер сравниваются с точными.

Таблица посчитана по точной матрице переходов с интервенцией.
Детальный баланс следует из построения; строка проверяет реализацию
$Q$. Предложение
по интервенции принимается в полтора раза чаще, спектральный зазор
немного больше. Из полного графа оно идёт к $G^*$ дольше: там $M$
не определён данными ($u_e=1$) и чаще предлагается к удалению.

### 4.5. Выбор графа""",
    ),
    (
        "code",
        """lab3_plots.plot_posterior(ROOT, rlc)
lab3_reporting.show_summary(results)""",
    ),
    (
        "code",
        """lab3_plots.plot_edge_frequencies(ROOT, results)
lab3_reporting.show_edge_report(results)
lab3_reporting.show_weight_sensitivity(rlc)""",
    ),
    (
        "markdown",
        r"""На обоих этапах и во всех задачах $G^*$ — граф, породивший данные.
В таблице $\Delta J_e$ — изменение $J$ при смене состояния ребра
в $G^*$. Каждое лишнее ребро проигрывает уже без интервенции
($\Delta J_e>0$), поэтому регуляризатор выбор не меняет.
Он умножает вес графа с лишним ребром
на $m^{-\lambda u_e/2}\approx m^{-\lambda/2}$: частота лишних рёбер
падает примерно на порядок, частота $G^*$ растёт.

Точки на рисунке и столбец «Точная частота» получены перебором; цепи
с ними согласны. Правило ЛР2, применённое к $G^*$ (для отсутствующего
ребра — к $G^*$ с этим ребром), согласно с выбором.

### 4.6. Выборка и аппроксимация выбранным графом""",
    ),
    (
        "code",
        """lab3_plots.plot_approximation(ROOT, results)""",
    ),
    (
        "markdown",
        r"""Точки — наблюдения, линия — модель выбранного графа, внизу остатки
в единицах $s(x)$. Ошибка на отложенной сетке ЛР2 более чем на порядок
меньше допуска $\tau$.

## Источники

- Metropolis N. и др. (1953). Equation of State Calculations by Fast
  Computing Machines. https://doi.org/10.1063/1.1699114
- Hastings W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains
  and Their Applications. https://doi.org/10.1093/biomet/57.1.97
- Madigan D., York J. (1995). Bayesian Graphical Models for Discrete Data.
  https://doi.org/10.2307/1403615
- Giudici P., Castelo R. (2003). Improving Markov Chain Monte Carlo Model
  Search for Data Mining. https://doi.org/10.1023/A:1020202028934
- Schwarz G. (1978). Estimating the Dimension of a Model.
  https://doi.org/10.1214/aos/1176344136
- Стрижов В. В. Прикладной регрессионный анализ, части 1 и 4;
  Diffusion of linear operators: the algorithm (2026).
  https://github.com/vadim-vic/causal-Inference
- Heiss A. (2021). Do-calculus adventures! Exploring the three rules
  of do-calculus.
  https://www.andrewheiss.com/blog/2021/09/07/do-calculus-backdoors/
- ЛР2: https://github.com/TeoSable/fda_lab2,
  ЛР1: https://github.com/KasiukVadim/iad_fall_2026
- Данные: Malitson (1965) https://doi.org/10.1364/JOSA.55.001205,
  Daimon, Masumura (2007) https://doi.org/10.1364/AO.46.003811;
  HTTP-адреса — в `data/provenance.json`.

Описание эксперимента и запуск — `docs/LAB3_GUIDE.md`.""",
    ),
)


def fill(text):
    """Insert the settings quoted in the text from the configuration."""
    values = {
        "@STEPS@": DEFAULT_CONFIG.steps,
        "@WEIGHT@": DEFAULT_CONFIG.stability_weight,
        "@SWAP@": DEFAULT_CONFIG.swap_probability,
        "@FLOOR@": DEFAULT_CONFIG.proposal_floor,
        "@ALPHA@": DEFAULT_CONFIG.intervention_alpha,
        "@REPEATS@": DEFAULT_CONFIG.intervention_repeats,
        "@BURN@": DEFAULT_CONFIG.burn_in,
    }
    for placeholder, value in values.items():
        text = text.replace(placeholder, f"{value:g}")
    return text


def sources():
    """Cell types and sources of the notebook, in order."""
    return [
        (kind, fill(source) if kind == "markdown" else source)
        for kind, source in CELLS
    ]


def build(path):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": KERNEL_DISPLAY_NAME,
        "language": "python",
        "name": KERNEL_NAME,
    }
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source)
        if kind == "markdown"
        else nbformat.v4.new_code_cell(source)
        for kind, source in sources()
    ]
    nbformat.write(notebook, path)


def update_text(path):
    """Replace the markdown of an executed notebook without rerunning it."""
    from check_lab3_claims import check_claims

    notebook = nbformat.read(path, as_version=4)
    expected = sources()
    code = [(c.cell_type, c.source) for c in notebook.cells]
    if [kind for kind, _ in code] != [kind for kind, _ in expected] or any(
        old != new for old, new in zip(code, expected) if new[0] == "code"
    ):
        raise SystemExit(
            "Code cells changed: rebuild and execute the notebook instead"
        )
    for cell, (_, source) in zip(notebook.cells, expected):
        cell.source = source
    nbformat.write(notebook, path)
    check_claims(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true")
    arguments = parser.parse_args()
    path = ROOT / NOTEBOOK_PATH
    if arguments.text_only:
        update_text(path)
        print(f"Updated the text of {path}; it agrees with the results")
    else:
        build(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
