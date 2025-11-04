"""
Knockout 시 성장과 목표 생산물 분석
유전자 knockout이 성장과 특정 대사물질 생산에 미치는 영향을 분석합니다.
전통적인 방법: 목표 생산물에 최소 플럭스를 설정하고 biomass를 최대화한 후 MOMA 수행
"""

from Simulator import Simulator
import cobra
import pandas as pd
import os
from datetime import datetime

def main():
    # 모델 로드
    print("모델 로딩 중...")
    sim = Simulator()
    model = cobra.io.load_model("iML1515")
    sim.load_cobra_model(model)
    
    # 목표 생산물 설정
    target_product = 'EX_succ_e'  # 숙신산
    min_production_flux = 0.1  # 최소 생산 플럭스 (mmol/gDW/hr)
    
    # 야생형 분석
    print("\n=== 야생형 분석 ===")
    
    # 1. 야생형 최대 성장
    _, wt_growth, wt_fluxes = sim.run_FBA()
    wt_production = wt_fluxes.get(target_product, 0)
    
    print(f"야생형 최대 성장: {wt_growth:.4f} hr⁻¹")
    print(f"  이때 {target_product} 생산: {wt_production:.4f} mmol/gDW/hr")
    
    # 2. 야생형에서 목표 생산물 최대 생산
    _, wt_max_prod, wt_max_fluxes = sim.run_FBA(new_objective=target_product, mode='max')
    wt_max_growth = wt_max_fluxes[sim.objective]
    print(f"야생형 최대 {target_product} 생산: {wt_max_prod:.4f} mmol/gDW/hr")
    print(f"  이때 성장: {wt_max_growth:.4f} hr⁻¹")
    
    # 3. 야생형 최소 생산 강제 + 성장 최대화 (MOMA 참조 상태)
    production_constraint = {target_product: (min_production_flux, 1000)}
    _, wt_constrained_growth, wt_constrained_fluxes = sim.run_FBA(
        flux_constraints=production_constraint
    )
    wt_constrained_production = wt_constrained_fluxes.get(target_product, 0)
    
    print(f"\n야생형 (최소 생산 {min_production_flux} 강제 시):")
    print(f"  성장: {wt_constrained_growth:.4f} hr⁻¹")
    print(f"  {target_product} 생산: {wt_constrained_production:.4f} mmol/gDW/hr")
    print(f"  → 이 상태를 MOMA의 참조 플럭스로 사용")
    
    # Knockout 테스트할 유전자 (간단한 테스트를 위해 2개만)
    genes_to_test = ['PGI', 'PFK']
    
    print(f"\n{'='*120}")
    print(f"=== {len(genes_to_test)}개 유전자 Knockout 후 성장 및 생산 분석 ===")
    print(f"{'='*120}")
    results = []
    
    for gene in genes_to_test:
        print(f"\n분석 중: {gene} knockout")
        
        knockout = {gene: (0, 0)}
        
        # MOMA로 knockout 예측 (최소 생산 강제된 야생형을 참조로 사용)
        status_moma, moma_distance, moma_fluxes = sim.run_MOMA(
            wild_flux=wt_constrained_fluxes,
            flux_constraints=knockout
        )
        
        if status_moma == 'optimal':
            # Knockout 후 성장 및 생산 (MOMA 예측)
            moma_growth = moma_fluxes[sim.objective]
            moma_production = moma_fluxes.get(target_product, 0)
            
            print(f"  MOMA 예측:")
            print(f"    성장: {moma_growth:.4f} hr⁻¹ (감소: {(1-moma_growth/wt_constrained_growth)*100:.1f}%)")
            print(f"    생산: {moma_production:.4f} mmol/gDW/hr")
            print(f"    대사 거리: {moma_distance:.4f}")
            
            # Knockout 후 최대 생산 능력 분석
            sim_ko = Simulator()
            sim_ko.load_cobra_model(model)
            
            # KO 후 최대 성장
            _, ko_max_growth, ko_max_growth_fluxes = sim_ko.run_FBA(
                flux_constraints=knockout
            )
            ko_max_growth_production = ko_max_growth_fluxes.get(target_product, 0)
            
            # KO 후 최대 생산
            _, ko_max_prod, ko_max_fluxes = sim_ko.run_FBA(
                new_objective=target_product,
                flux_constraints=knockout,
                mode='max'
            )
            ko_max_prod_growth = ko_max_fluxes[sim_ko.objective]
            
            # KO 후 최소 생산 강제 + 성장 최대화
            ko_constraints = knockout.copy()
            ko_constraints[target_product] = (min_production_flux, 1000)
            _, ko_constrained_growth, ko_constrained_fluxes = sim_ko.run_FBA(
                flux_constraints=ko_constraints
            )
            ko_constrained_production = ko_constrained_fluxes.get(target_product, 0)
            
            print(f"  FBA 분석:")
            print(f"    최대 성장: {ko_max_growth:.4f} hr⁻¹ (생산: {ko_max_growth_production:.4f})")
            print(f"    최대 생산: {ko_max_prod:.4f} mmol/gDW/hr (성장: {ko_max_prod_growth:.4f})")
            print(f"    최소 생산 강제 시 성장: {ko_constrained_growth:.4f} hr⁻¹ (생산: {ko_constrained_production:.4f})")
            
            results.append({
                '유전자': gene,
                
                # 야생형
                'WT_최대성장': wt_growth,
                'WT_최대성장시_생산': wt_production,
                'WT_최대생산': wt_max_prod,
                'WT_최대생산시_성장': wt_max_growth,
                'WT_제약성장': wt_constrained_growth,
                'WT_제약성장시_생산': wt_constrained_production,
                
                # MOMA 예측
                'MOMA_성장': moma_growth,
                'MOMA_생산': moma_production,
                'MOMA_성장감소율(%)': (1 - moma_growth/wt_constrained_growth) * 100,
                'MOMA_대사거리': moma_distance,
                
                # Knockout FBA
                'KO_최대성장': ko_max_growth,
                'KO_최대성장시_생산': ko_max_growth_production,
                'KO_최대성장_감소율(%)': (1 - ko_max_growth/wt_growth) * 100,
                'KO_최대생산': ko_max_prod,
                'KO_최대생산시_성장': ko_max_prod_growth,
                'KO_최대생산_변화율(%)': ((ko_max_prod - wt_max_prod) / wt_max_prod * 100) if wt_max_prod > 1e-6 else 0,
                'KO_제약성장': ko_constrained_growth,
                'KO_제약성장시_생산': ko_constrained_production,
                'KO_제약성장_감소율(%)': (1 - ko_constrained_growth/wt_constrained_growth) * 100,
                
                '상태': 'Success'
            })
        else:
            print(f"  상태: Infeasible (치명적 knockout)")
            
            results.append({
                '유전자': gene,
                'WT_최대성장': wt_growth,
                'WT_최대성장시_생산': wt_production,
                'WT_최대생산': wt_max_prod,
                'WT_최대생산시_성장': wt_max_growth,
                'WT_제약성장': wt_constrained_growth,
                'WT_제약성장시_생산': wt_constrained_production,
                'MOMA_성장': 0,
                'MOMA_생산': 0,
                'MOMA_성장감소율(%)': 100,
                'MOMA_대사거리': float('nan'),
                'KO_최대성장': 0,
                'KO_최대성장시_생산': 0,
                'KO_최대성장_감소율(%)': 100,
                'KO_최대생산': 0,
                'KO_최대생산시_성장': 0,
                'KO_최대생산_변화율(%)': -100,
                'KO_제약성장': 0,
                'KO_제약성장시_생산': 0,
                'KO_제약성장_감소율(%)': 100,
                '상태': 'Infeasible'
            })
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    
    # 결과 출력
    print("\n" + "="*120)
    print("=== 야생형 기준값 ===")
    print("="*120)
    print(f"최대 성장: {wt_growth:.4f} hr⁻¹ (생산: {wt_production:.4f} mmol/gDW/hr)")
    print(f"최대 생산: {wt_max_prod:.4f} mmol/gDW/hr (성장: {wt_max_growth:.4f} hr⁻¹)")
    print(f"제약 조건 (최소 생산 {min_production_flux}): 성장 {wt_constrained_growth:.4f} hr⁻¹, 생산 {wt_constrained_production:.4f} mmol/gDW/hr")
    
    print("\n" + "="*120)
    print("=== MOMA 예측: Knockout 후 즉각 반응 ===")
    print("="*120)
    print(df[['유전자', 'MOMA_성장', 'MOMA_생산', 'MOMA_성장감소율(%)', 'MOMA_대사거리', '상태']].to_string(index=False))
    
    print("\n" + "="*120)
    print("=== FBA 분석: Knockout 후 최대 성장 ===")
    print("="*120)
    print(df[['유전자', 'KO_최대성장', 'KO_최대성장시_생산', 'KO_최대성장_감소율(%)', '상태']].to_string(index=False))
    
    print("\n" + "="*120)
    print("=== FBA 분석: Knockout 후 최대 생산 능력 ===")
    print("="*120)
    print(df[['유전자', 'KO_최대생산', 'KO_최대생산시_성장', 'KO_최대생산_변화율(%)', '상태']].to_string(index=False))
    
    print("\n" + "="*120)
    print("=== FBA 분석: Knockout 후 제약 조건 하 성장 ===")
    print("="*120)
    print(df[['유전자', 'KO_제약성장', 'KO_제약성장시_생산', 'KO_제약성장_감소율(%)', '상태']].to_string(index=False))
    
    # CSV 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 전체 결과 저장
    csv_filename = f"{output_dir}/knockout_analysis_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n전체 결과가 저장되었습니다: {csv_filename}")
    
    # 요약 결과 저장
    summary_df = df[['유전자', 
                     'WT_최대성장', 'WT_최대생산',
                     'MOMA_성장', 'MOMA_생산', 'MOMA_성장감소율(%)',
                     'KO_최대성장', 'KO_최대성장_감소율(%)',
                     'KO_최대생산', 'KO_최대생산_변화율(%)',
                     '상태']].copy()
    
    summary_filename = f"{output_dir}/knockout_summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
    print(f"요약 결과가 저장되었습니다: {summary_filename}")
    
    # 생산 증가 전략 제안
    print("\n" + "="*120)
    print("=== 생산 최적화 전략 제안 ===")
    print("="*120)
    
    viable_knockouts = df[(df['상태'] == 'Success') & (df['KO_최대성장'] > 0.1 * wt_growth)]
    
    if len(viable_knockouts) > 0:
        # 최대 생산이 증가한 경우
        improved = viable_knockouts[viable_knockouts['KO_최대생산'] > wt_max_prod * 1.01]
        if len(improved) > 0:
            print(f"\n✓ 최대 생산 증가 knockout 후보:")
            for _, row in improved.iterrows():
                increase = row['KO_최대생산_변화율(%)']
                print(f"  • {row['유전자']}: 최대 생산 {increase:.1f}% 증가")
        else:
            print(f"\n✗ 최대 생산이 증가하는 knockout 후보가 없습니다.")
        
        # 제약 조건 하에서 성장이 개선된 경우
        better_constrained = viable_knockouts[viable_knockouts['KO_제약성장'] > wt_constrained_growth * 1.01]
        if len(better_constrained) > 0:
            print(f"\n✓ 제약 조건 하 성장 개선 knockout 후보:")
            for _, row in better_constrained.iterrows():
                print(f"  • {row['유전자']}: 제약 성장 {row['KO_제약성장_감소율(%)']:.1f}% 개선")
        else:
            print(f"\n✗ 제약 조건 하에서 성장이 개선되는 knockout 후보가 없습니다.")
    else:
        print("\n✗ 실행 가능한 knockout 후보가 없습니다.")

if __name__ == "__main__":
    main()

