def test_torch_geometric_data():
    """PyTorch Geometric Data 객체 처리 테스트"""
    print("\n🧪 PyTorch Geometric Data 객체 테스트...")
    
    train_dir = "data/processed/binding/train"
    
    if not os.path.exists(train_dir):
        print("❌ 훈련 데이터 디렉토리가 없습니다")
        return False
    
    files = [f for f in os.listdir(train_dir) if f.endswith('.pt')][:5]
    
    if not files:
        print("❌ 데이터 파일이 없습니다")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 사용 디바이스: {device}")
    
    success_count = 0
    
    for i, filename in enumerate(files):
        filepath = os.path.join(train_dir, filename)
        print(f"\n📁 테스트 {i+1}/{len(files)}: {filename}")
        
        try:
            data = torch.load(filepath, weights_only=False)
            
            # PyTorch Geometric Data 객체 처리
            if hasattr(data, 'x') and hasattr(data, 'pos') and hasattr(data, 'edge_index'):
                x = data.x.float().to(device)
                pos = data.pos.float().to(device)
                edge_index = data.edge_index.long().to(device)
                
                # 라벨 처리
                if hasattr(data, 'y'):
                    label = data.y.float().to(device)
                    if label.dim() > 1:
                        label = label.squeeze()
                else:
                    label = torch.tensor(5.0).to(device)  # 기본값
                
                print(f"   노드: {x.shape[0]}, 에지: {edge_index.shape[1]}, 특징차원: {x.shape[1]}")
                print(f"   라벨: {label.item():.3f}")
                
                # 적응적 모델 크기
                num_nodes = x.shape[0]
                if num_nodes > 2000:
                    hidden_size = 32
                    layers = 1
                elif num_nodes > 1000:
                    hidden_size = 64
                    layers = 2
                else:
                    hidden_size = 128
                    layers = 3
                
                # 모델 생성 및 테스트
                model = EGNN(
                    in_dim=x.size(1),
                    hidden=hidden_size,
                    layers=layers,
                    dropout=0.1
                ).to(device)
                
                model.eval()
                with torch.no_grad():
                    output = model(x, pos, edge_index)
                    pred_val = output.item() if output.numel() == 1 else output[0].item()
                    
                    print(f"   ✅ 예측: {pred_val:.3f}, 실제: {label.item():.3f}")
                    success_count += 1
                    
            else:
                print(f"   ❌ 필수 속성 누락: x, pos, edge_index")
                
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    print(f"\n📊 PyG Data 테스트 결과: {success_count}/{len(files)} 성공")
    return success_count > 0

def test_edge_case_handling():
    """엣지 케이스 처리 테스트"""
    print("\n🧪 엣지 케이스 테스트...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_cases = []
    
    # 1. 매우 작은 분자
    small_mol = {
        'x': torch.randn(5, 8),
        'pos': torch.randn(5, 3),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        'name': '작은 분자 (5개 노드)'
    }
    test_cases.append(small_mol)
    
    # 2. 매우 큰 분자 (메모리 제한 테스트)
    large_mol = {
        'x': torch.randn(5000, 8),
        'pos': torch.randn(5000, 3),
        'edge_index': torch.randint(0, 5000, (2, 25000)),
        'name': '큰 분자 (5000개 노드)'
    }
    test_cases.append(large_mol)
    
    # 3. 에지가 없는 분자
    no_edge_mol = {
        'x': torch.randn(10, 8),
        'pos': torch.randn(10, 3),
        'edge_index': torch.empty((2, 0), dtype=torch.long),
        'name': '에지 없는 분자'
    }
    test_cases.append(no_edge_mol)
    
    # 4. 극값을 가진 분자
    extreme_mol = {
        'x': torch.randn(100, 8) * 100,  # 큰 특징값
        'pos': torch.randn(100, 3) * 1000,  # 큰 좌표값
        'edge_index': torch.randint(0, 100, (2, 500)),
        'name': '극값 분자'
    }
    test_cases.append(extreme_mol)
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📁 테스트 {i+1}: {test_case['name']}")
        
        try:
            x = test_case['x'].to(device)
            pos = test_case['pos'].to(device)
            edge_index = test_case['edge_index'].to(device)
            
            # 자기 연결 제거
            if edge_index.size(1) > 0:
                mask = edge_index[0] != edge_index[1]
                edge_index = edge_index[:, mask]
            
            print(f"   노드: {x.shape[0]}, 에지: {edge_index.shape[1]}")
            
            # 적응적 모델 설정
            num_nodes = x.shape[0]
            if num_nodes > 3000:
                hidden, layers = 16, 1
            elif num_nodes > 1000:
                hidden, layers = 32, 2
            else:
                hidden, layers = 64, 3
            
            model = EGNN(
                in_dim=x.size(1),
                hidden=hidden,
                layers=layers,
                dropout=0.1
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(x, pos, edge_index)
                pred_val = output.item() if output.numel() == 1 else output[0].item()
                
                print(f"   ✅ 예측값: {pred_val:.3f}")
                success_count += 1
                
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    print(f"\n📊 엣지 케이스 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count > 0

def test_memory_efficiency():
    """메모리 효율성 테스트"""
    print("\n🧪 메모리 효율성 테스트...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA가 없어 메모리 테스트를 건너뜁니다")
        return False
    
    device = torch.device('cuda')
    
    # 초기 메모리 사용량
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    print(f"📊 초기 GPU 메모리: {initial_memory / 1024**2:.1f} MB")
    
    # 점진적으로 큰 분자들 테스트
    node_sizes = [100, 500, 1000, 2000, 3000]
    memory_usage = []
    
    for num_nodes in node_sizes:
        try:
            torch.cuda.empty_cache()
            
            num_edges = min(num_nodes * 5, 50000)  # 에지 수 제한
            
            x = torch.randn(num_nodes, 8).to(device)
            pos = torch.randn(num_nodes, 3).to(device)
            edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
            
            # 자기 연결 제거
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
            # 적응적 모델
            if num_nodes > 2000:
                hidden, layers = 32, 1
            elif num_nodes > 1000:
                hidden, layers = 64, 2
            else:
                hidden, layers = 128, 3
            
            model = EGNN(
                in_dim=8,
                hidden=hidden,
                layers=layers,
                dropout=0.1
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(x, pos, edge_index)
            
            current_memory = torch.cuda.memory_allocated()
            memory_used = (current_memory - initial_memory) / 1024**2
            memory_usage.append(memory_used)
            
            print(f"   노드 {num_nodes:4d}: {memory_used:6.1f} MB")
            
            # 메모리 정리
            del x, pos, edge_index, model, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ 노드 {num_nodes}에서 실패: {e}")
            memory_usage.append(None)
    
    # 메모리 효율성 분석
    valid_usage = [m for m in memory_usage if m is not None]
    if len(valid_usage) >= 2:
        growth_rate = (valid_usage[-1] - valid_usage[0]) / (node_sizes[len(valid_usage)-1] - node_sizes[0])
        print(f"📈 메모리 증가율: {growth_rate:.3f} MB/노드")
        
        if growth_rate < 0.01:  # 노드당 0.01MB 미만
            print("✅ 메모리 효율적")
            return True
        else:
            print("⚠️ 메모리 사용량이 높음")
            return False
    
    return len(valid_usage) > 0

def test_gradient_flow():
    """그래디언트 흐름 테스트"""
    print("\n🧪 그래디언트 흐름 테스트...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 테스트 데이터 생성
    num_nodes = 500
    x = torch.randn(num_nodes, 8, requires_grad=True).to(device)
    pos = torch.randn(num_nodes, 3, requires_grad=True).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 2000)).to(device)
    target = torch.tensor(5.0).to(device)
    
    # 자기 연결 제거
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    model = EGNN(
        in_dim=8,
        hidden=64,
        layers=2,
        dropout=0.1
    ).to(device)
    
    # 순전파
    model.train()
    output = model(x, pos, edge_index)
    loss = F.mse_loss(output, target.unsqueeze(0))
    
    print(f"📊 손실값: {loss.item():.6f}")
    
    # 역전파
    loss.backward()
    
    # 그래디언트 체크
    total_grad_norm = 0
    param_count = 0
    nan_count = 0
    zero_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            
            if torch.isnan(param.grad).any():
                nan_count += 1
                print(f"   ❌ NaN 그래디언트: {name}")
            elif grad_norm < 1e-8:
                zero_count += 1
                print(f"   ⚠️ 0에 가까운 그래디언트: {name} (norm: {grad_norm:.2e})")
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
    
    print(f"📈 그래디언트 통계:")
    print(f"   평균 노름: {avg_grad_norm:.6f}")
    print(f"   NaN 개수: {nan_count}/{param_count}")
    print(f"   0에 가까운 개수: {zero_count}/{param_count}")
    
    success = nan_count == 0 and avg_grad_norm > 1e-8
    print(f"{'✅' if success else '❌'} 그래디언트 흐름 {'정상' if success else '비정상'}")
    
    return success

def test_numerical_stability():
    """수치적 안정성 테스트"""
    print("\n🧪 수치적 안정성 테스트...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 다양한 스케일의 입력으로 테스트
    test_configs = [
        {'scale': 0.01, 'name': '매우 작은 값'},
        {'scale': 1.0, 'name': '정상 값'},
        {'scale': 10.0, 'name': '큰 값'},
        {'scale': 100.0, 'name': '매우 큰 값'}
    ]
    
    success_count = 0
    
    for config in test_configs:
        print(f"\n📁 테스트: {config['name']} (스케일: {config['scale']})")
        
        try:
            scale = config['scale']
            num_nodes = 200
            
            x = torch.randn(num_nodes, 8) * scale
            pos = torch.randn(num_nodes, 3) * scale
            edge_index = torch.randint(0, num_nodes, (2, 800))
            
            # 자기 연결 제거
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
            x = x.to(device)
            pos = pos.to(device)
            edge_index = edge_index.to(device)
            
            model = EGNN(
                in_dim=8,
                hidden=32,
                layers=2,
                dropout=0.1
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(x, pos, edge_index)
                
                # NaN/Inf 체크
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"   ❌ 출력에 NaN/Inf 발생")
                else:
                    pred_val = output.item() if output.numel() == 1 else output[0].item()
                    print(f"   ✅ 안정적 출력: {pred_val:.3f}")
                    success_count += 1
                    
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    print(f"\n📊 수치적 안정성 테스트: {success_count}/{len(test_configs)} 성공")
    return success_count == len(test_configs)

def test_model_consistency():
    """모델 일관성 테스트"""
    print("\n🧪 모델 일관성 테스트...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 동일한 입력으로 여러 번 실행
    num_nodes = 300
    x = torch.randn(num_nodes, 8).to(device)
    pos = torch.randn(num_nodes, 3).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 1200)).to(device)
    
    # 자기 연결 제거
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    model = EGNN(
        in_dim=8,
        hidden=64,
        layers=2,
        dropout=0.0  # 드롭아웃 비활성화
    ).to(device)
    
    model.eval()
    outputs = []
    
    # 10번 실행
    for i in range(10):
        with torch.no_grad():
            output = model(x, pos, edge_index)
            pred_val = output.item() if output.numel() == 1 else output[0].item()
            outputs.append(pred_val)
    
    # 일관성 체크
    std_dev = np.std(outputs)
    mean_val = np.mean(outputs)
    
    print(f"📊 예측값 통계:")
    print(f"   평균: {mean_val:.6f}")
    print(f"   표준편차: {std_dev:.8f}")
    print(f"   범위: {min(outputs):.6f} ~ {max(outputs):.6f}")
    
    # 표준편차가 매우 작아야 함 (일관성)
    is_consistent = std_dev < 1e-6
    print(f"{'✅' if is_consistent else '❌'} 모델 {'일관성 있음' if is_consistent else '불일치'}")
    
    return is_consistent

def benchmark_performance():
    """성능 벤치마크 테스트"""
    print("\n🧪 성능 벤치마크 테스트...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 다양한 크기의 분자로 성능 측정
    sizes = [100, 500, 1000, 2000]
    results = []
    
    for num_nodes in sizes:
        print(f"\n📁 벤치마크: {num_nodes}개 노드")
        
        try:
            num_edges = min(num_nodes * 5, 20000)
            
            x = torch.randn(num_nodes, 8).to(device)
            pos = torch.randn(num_nodes, 3).to(device)
            edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
            
            # 자기 연결 제거
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
            # 적응적 모델
            if num_nodes > 1500:
                hidden, layers = 32, 1
            elif num_nodes > 800:
                hidden, layers = 64, 2
            else:
                hidden, layers = 128, 3
            
            model = EGNN(
                in_dim=8,
                hidden=hidden,
                layers=layers,
                dropout=0.1
            ).to(device)
            
            model.eval()
            
            # 워밍업
            with torch.no_grad():
                _ = model(x, pos, edge_index)
            
            # 시간 측정
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            num_runs = 10
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(x, pos, edge_index)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            throughput = num_nodes / avg_time
            
            results.append({
                'nodes': num_nodes,
                'edges': edge_index.shape[1],
                'time': avg_time,
                'throughput': throughput
            })
            
            print(f"   평균 시간: {avg_time:.4f}초")
            print(f"   처리량: {throughput:.0f} 노드/초")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    # 성능 요약
    if results:
        print(f"\n📊 성능 벤치마크 요약:")
        for result in results:
            print(f"   {result['nodes']:4d} 노드: {result['time']:.4f}초 ({result['throughput']:.0f} 노드/초)")
        
        # 스케일링 효율성
        if len(results) >= 2:
            small_throughput = results[0]['throughput']
            large_throughput = results[-1]['throughput']
            efficiency = large_throughput / small_throughput
            
            print(f"📈 스케일링 효율성: {efficiency:.2f}")
            return efficiency > 0.1  # 최소 10% 효율성 유지
    
    return len(results) > 0

def main():
    """메인 테스트 함수 (확장된 버전)"""
    print("🚀 EGNN 모델 종합 테스트 시작\n")
    
    import time
    
    test_results = {}
    
    # 1. 데이터 구조 분석
    inspect_data_structure()
    
    # 2. PyTorch Geometric Data 테스트
    print("\n" + "="*60)
    test_results['pyg_data'] = test_torch_geometric_data()
    
    # 3. 실제 단일 샘플 테스트 (기존)
    print("\n" + "="*60)
    real_data = load_real_sample_data()
    
    if real_data:
        test_results['single_sample'] = test_model_with_data(real_data)
    else:
        print("❌ 실제 데이터 로드 실패, 더미 데이터로 대체")
        dummy_data = create_dummy_data()
        test_results['single_sample'] = test_model_with_data(dummy_data)
    
    # 4. 배치 처리 테스트 (기존)
    print("\n" + "="*60)
    test_results['batch_processing'] = test_batch_processing()
    
    # 5. 엣지 케이스 테스트
    print("\n" + "="*60)
    test_results['edge_cases'] = test_edge_case_handling()
    
    # 6. 메모리 효율성 테스트
    print("\n" + "="*60)
    test_results['memory_efficiency'] = test_memory_efficiency()
    
    # 7. 그래디언트 흐름 테스트
    print("\n" + "="*60)
    test_results['gradient_flow'] = test_gradient_flow()
    
    # 8. 수치적 안정성 테스트
    print("\n" + "="*60)
    test_results['numerical_stability'] = test_numerical_stability()
    
    # 9. 모델 일관성 테스트
    print("\n" + "="*60)
    test_results['model_consistency'] = test_model_consistency()
    
    # 10. 성능 벤치마크
    print("\n" + "="*60)
    test_results['performance'] = benchmark_performance()
    
    # 최종 결과
    print(f"\n" + "="*60)
    print(f"📋 종합 테스트 결과:")
    
    passed = 0
    total = 0
    
    for test_name, result in test_results.items():
        status = '✅ 성공' if result else '❌ 실패'
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
        total += 1
    
    success_rate = (passed / total) * 100
    print(f"\n📊 전체 성공률: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print(f"\n🎉 대부분의 테스트 통과! EGNN 모델이 안정적으로 작동합니다.")
        if test_results.get('pyg_data', False):
            print(f"   ✨ PyTorch Geometric 데이터 처리 가능")
        if test_results.get('memory_efficiency', False):
            print(f"   💾 메모리 효율적")
        if test_results.get('numerical_stability', False):
            print(f"   🔢 수치적으로 안정적")
    else:
        print(f"\n⚠️ 일부 테스트 실패 ({success_rate:.1f}% 성공). 세부 사항을 확인해주세요.")

if __name__ == "__main__":
    main()