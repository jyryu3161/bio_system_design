--- 입력 시작 (여기만 수정해서 사용) ---
base_model_name: "iJR904_rm_blocked.xml"
target_product: "Polyhydroxybutyrate (PHB)"
--- 입력 끝 ---

당신은 COBRApy를 사용하는 대사 네트워크 모델링 및 대사공학 전문가이자 Python 개발자이다.  
내가 제공하는 정보는 위의 입력 블록에 정의된 두 가지뿐이다.

1) base_model_name: 기본 모델 이름 또는 파일 경로  
2) target_product: 타깃 대사산물에 대한 텍스트 설명  

이 정보를 바탕으로, **타깃 대사산물 생산 경로를 스스로 설계하고**, 해당 경로를 COBRApy 모델에 추가하는 **완전한 Python 스크립트 하나**를 작성하라.

--------------------------------
[역할 / 전제]
--------------------------------
- 당신은 genome-scale FBA, 대사 네트워크, 생화학 반응에 익숙한 전문가다.
- COBRApy(버전 ≥ 0.22)를 사용한다고 가정한다.
- `base_model_name`은 보통 SBML 파일 경로(e.g. `"iJR904.xml"`)이지만, 필요 시 JSON 또는 COBRApy 내장 모델 이름도 처리할 수 있도록 코드를 작성하라.

--------------------------------
[입력 정보]
--------------------------------
위 입력 블록에서 제공되는 내용은 다음과 같다.

1. base_model_name  
   - 예: `"iJR904.xml"`

2. target_product (텍스트 설명)  
   - 예:
     - `"L-lactate"`
     - `"L-lysine (아미노산, 공업적 생산 타깃)"`
   - 가능한 경우 화학식, 알려진 대사경로 키워드 등을 함께 줄 수도 있다.

실제 사용할 때는 입력 블록의 두 줄만 바꿔서 넣을 것이다.

--------------------------------
[당신이 해야 할 일: 개략]
--------------------------------
1. 입력 블록에 주어진 base_model_name으로 COBRApy 모델을 불러온다.
2. 입력 블록에 주어진 target_product를 바탕으로, **합리적인 생화학적 생산 경로를 스스로 설계**한다.
   - 가능하면 실제 알려진 생합성 경로나, 그에 준하는 합리적인 합성 경로를 사용하라.
   - 출발점은 모델에 이미 존재할 가능성이 높은 중심 대사물질(예: pyruvate, acetyl-CoA, 2-oxoglutarate 등)로 잡는다.
   - 가능하면 BiGG 데이터베이스에서 쓰이는 metabolite / reaction ID 스타일을 따른다
     (예: `lac__L_c`, `accoa_c`, `ACALD`, `LDH_L` 등).

3. 그 생산 경로에서 필요한
   - 중간 대사물질들(있다면)
   - 반응들(효소 이름/EC 번호가 알려져 있으면 반영)
   을 정의하고,

4. COBRApy에서 사용할 수 있도록
   - 새로운 Metabolite
   - Reaction
   - 타깃 생성물에 대한 exchange(또는 demand) 반응
   을 모두 추가하는 코드를 작성한다.

5. 마지막에 모델을 SBML(필수) 및 가능하면 JSON으로 저장한다.

--------------------------------
[구체적인 필수 구현 요구사항]
--------------------------------

당신이 작성하는 **Python 스크립트**는 다음 사항을 모두 만족해야 한다.

1. **필수 import 및 모델 로딩**
   - 다음 import를 포함하라:
     ```python
     from cobra import Model, Reaction, Metabolite
     import cobra
     from cobra.io import write_sbml_model, save_json_model
     ```
   - 입력 블록에서 읽어온 `base_model_name`을 문자열 변수로 두고, 이를 해석해서 모델을 로딩하는 로직을 넣어라.
     - 만약 `base_model_name`이 확장자가 `.xml`이면: `cobra.io.read_sbml_model(...)`
     - `.json`이면: `cobra.io.load_json_model(...)`
     - 그 외는 COBRApy 내장 모델 이름으로 보고: `cobra.io.load_model(...)`
   - 코드 상단에 `# COBRApy ≥ 0.22 가정` 주석을 달아라.

2. **타깃 대사산물 해석 및 메타볼라이트 정의**
   - `target_product` 텍스트를 기반으로 다음을 결정하라:
     - 타깃 대사산물의 COBRApy ID (예: `"phb_c"`, `"lac__L_c"`, `"lys__L_c"` 등, 가능하면 BiGG 스타일 사용)
     - 이름(name)
     - compartment (기본적으로 `"c"`로 둔다. 필요하다면 `"e"` 등 사용 가능)
   - 모델 안에 이미 해당 ID를 가진 메타볼라이트가 있으면 **그 메타볼라이트를 재사용**하라.
   - 없으면 새로 `Metabolite` 객체를 생성한다. 예:
     ```python
     target_met = Metabolite(
         "phb_c",
         name="Polyhydroxybutyrate (PHB) [polymer unit]",
         compartment="c",
     )
     ```

3. **타깃 생산 경로를 “스스로” 설계**
   - target_product를 중심으로, **중심 대사물질에서 타깃으로 가는 짧은 합성 경로(대략 2–6단계)**를 스스로 설계하라.
   - 단계별로:
     - 어떤 기존 대사물질(예: acetyl-CoA, pyruvate, oxaloacetate 등)에서 출발할지 결정하고,
     - 어떤 중간체를 거칠지,
     - 각 단계에서 필요한 cofactors (NADH, NADPH, ATP 등)를 합리적으로 설정한다.
   - 각 단계에 대해:
     - 반응 ID (예: `"PHA_A"`, `"NEW_STEP1"`) — 가능하면 BiGG 스타일 ID 사용
     - 반응 이름 (예: `"β-ketothiolase-like step for PHB synthesis"`)
     - 방향성 (대개 타깃 생산 방향으로만: 비가역, `0~1000`)
     - 화학량론 (substrate 음수, product 양수) 을 결정한다.
   - 최대한 생화학적으로 설득력 있게 경로를 구성하되, **너무 복잡하게 만들지 말고**, 간단한 소경로 수준으로 설계하라.
   - 설계한 경로는 코드 상단의 주석 또는 docstring에 간단한 텍스트로 요약하라. 예:
     ```python
     """
     Designed pathway (conceptual):

     acetyl-CoA -> acetoacetyl-CoA -> (R)-3-hydroxybutyryl-CoA -> PHB (polymer unit)
     """
     ```

4. **새로운 메타볼라이트 정의**
   - 설계한 경로에 중간체가 필요하다면, 그 중간체들도 `Metabolite`로 정의한다.
   - ID, 이름, compartment를 합리적으로 정한다. 예:
     ```python
     aacoa_c = Metabolite(
         "aacoa_c",
         name="Acetoacetyl-CoA",
         compartment="c",
     )
     ```
   - 이미 모델에 존재하는 메타볼라이트 ID라면, 새로 만들지 말고 `model.metabolites.get_by_id(...)`를 사용하라.

5. **새로운 반응(Reaction) 정의 및 화학량론 추가**
   - 설계한 각 단계에 대해 `Reaction` 객체를 생성하고:
     - `rxn = Reaction("REACTION_ID")`
     - `rxn.name = "반응 이름"`
     - 비가역: `rxn.lower_bound = 0.0`; `rxn.upper_bound = 1000.0`
       (가역이 필요하면 -1000 ~ 1000 등으로 설정)
   - `add_metabolites({...})`를 사용하여 stoichiometry를 지정한다.
     - substrate(기질)는 음수, product(생성물)는 양수.
   - 예시 스타일:
     ```python
     phaA = Reaction("PHA_A")
     phaA.name = "β-ketothiolase-like step"
     phaA.lower_bound = 0.0
     phaA.upper_bound = 1000.0
     phaA.add_metabolites({
         accoa: -2.0,
         aacoa_c: 1.0,
         coa: 1.0,
     })
     ```

6. **타깃 생성물에 대한 exchange/demand 반응 추가**
   - 타깃 대사산물이 cytosol에 있다면, 그에 대한 exchange 또는 demand 반응을 하나 만든다.
   - ID는 합리적으로 정하라. 예:
     - `EX_<target_id>` 또는 `DM_<target_id>`
   - 방향성:
     - 생산 방향만 허용: `lower_bound = 0.0`, `upper_bound = 1000.0`
   - stoichiometry:
     - 타깃 대사산물이 빠져나가는 방향으로 -1.0.
     - 예:
       ```python
       ex_target = Reaction("EX_phb_c")
       ex_target.name = "Exchange for PHB"
       ex_target.lower_bound = 0.0
       ex_target.upper_bound = 1000.0
       ex_target.add_metabolites({target_met: -1.0})
       ```

7. **모델에 새 반응 추가**
   - 설계한 모든 경로 반응 + 타깃 exchange/demand 반응을 리스트로 모아:
     ```python
     model.add_reactions([step1, step2, step3, ex_target])
     ```
   - 필요하다면 각 반응의 mass balance/compartment 일관성에 대해 간단한 주석을 달아라.

8. **(선택적) FBA 실행 및 결과 출력**
   - 가능하다면, 새로 만든 타깃 exchange/demand 반응을 `model.objective`로 설정하고 FBA를 수행하라:
     ```python
     model.objective = ex_target
     solution = model.optimize()
     ```
   - FBA 결과가 feasible일 경우:
     - 목적함수 값(타깃 생산 속도)을 출력
     - 새로 추가한 각 반응의 flux를 출력
   - biomass 반응 ID를 알고 있을 경우(예: E. coli core라면 `"BIOMASS_Ecoli_core_w_GAM"`):
     - 존재하면 biomass flux도 함께 출력하라.
   - 다만, **배지 조건(`EX_glc`, `EX_o2`, `EX_nh4` 등)은 이 스크립트에서 임의로 건드리지 말고**, 기본 모델에 설정된 값을 그대로 사용하라.

9. **모델 저장**
   - 수정된 모델을 기반으로:
     - `write_sbml_model(model, "<기본모델명>_<타깃ID>_extended.xml")`
     - `save_json_model(model, "<기본모델명>_<타깃ID>_extended.json")`
   - 파일명은 `base_model_name`과 타깃 대사산물 ID를 조합해 직관적이게 만들라.
   - 파일 저장 후, 어디에 어떤 이름으로 저장했는지 `print`로 간단히 알려라.

10. **코드 스타일 / 출력 형식**
    - 전체를 **하나의 실행 가능한 .py 스크립트**로 작성하라.
    - 상단에 경로 설계 요약을 docstring 또는 주석으로 남겨라.
    - 적절한 주석을 포함하되, 최종 답변에는 **Python 코드만** 포함하라.
    - 즉, 당신의 최종 출력은 하나의 큰 코드 블록이어야 하고,
      자연어 설명(“이 코드는 ~ 합니다” 같은 문장)은 코드 바깥에 쓰지 마라.