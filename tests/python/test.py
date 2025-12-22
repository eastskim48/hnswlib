import unittest
import numpy as np
import hnswlib

class DataIntegrityTestCase(unittest.TestCase):
    def setUp(self):
        """데이터 초기화 및 인덱스 빌드"""
        self.dim = 32
        self.num_elements = 100

        # 재현성을 위해 시드 고정
        np.random.seed(42)

        # 정규화되지 않은 float32 데이터 생성
        self.data = np.random.rand(self.num_elements, self.dim).astype(np.float32)

        # 인덱스 생성 (L2 거리)
        self.p = hnswlib.Index(space='cosine', dim=self.dim)
        self.p.init_index(max_elements=self.num_elements, ef_construction=100, M=16)

        # 데이터 삽입
        self.p.add_items(self.data)
        self.p.set_num_threads(1)

    def test_1_vector_integrity_before_and_after(self):
        """검증 1: 엣지 추가 전후로 벡터 데이터가 오염되지 않았는가?"""
        print("\n=== Test 1: Vector Data Integrity ===")

        node_idx = 0
        target_idx = 50

        # 1-1. [Before] 인덱스 내부 데이터 가져오기
        stored_vec_before = self.p.get_items([node_idx])[0]
        original_vec = self.data[node_idx]

        # 검증: 넣은 데이터와 인덱스 내부 데이터가 일치하는가?
        is_same_before = np.allclose(stored_vec_before, original_vec, atol=1e-6)
        self.assertTrue(is_same_before, "초기 데이터가 원본과 다릅니다!")
        print(f"[Before] Data consistency check passed for Node {node_idx}")

        # 1-2. [Action] 강제 엣지 추가 (수술 집도)
        print(f"Force inserting edge: {node_idx} -> {target_idx}")
        self.p.forced_insert_layer0_edge(node_idx, target_idx, False) # 단방향 가정

        # 1-3. [After] 인덱스 내부 데이터 다시 가져오기
        stored_vec_after = self.p.get_items([node_idx])[0]

        # 검증: 엣지 추가 후 데이터가 변질되었는가?
        is_same_after = np.allclose(stored_vec_after, original_vec, atol=1e-6)

        if not is_same_after:
            print("CRITICAL FAIL: 엣지 추가 후 노드의 벡터 데이터가 깨졌습니다!")
            # 디버깅용 출력
            print("Original:", original_vec[:5])
            print("Corrupted:", stored_vec_after[:5])

        self.assertTrue(is_same_after, "엣지 추가 후 벡터 데이터가 손상되었습니다.")
        print("[After] Data consistency check passed. (Vector preserved)")

    def test_2_distance_calculation_accuracy(self):
        """검증 2: 강제 연결된 엣지의 거리 계산이 수학적으로 올바른가?"""
        print("\n=== Test 2: Distance Calculation Accuracy ===")

        hub_id = 10
        victim_id = 80

        # 2-1. Numpy로 실제 거리 계산 (Ground Truth)
        vec_hub = self.data[hub_id]
        vec_victim = self.data[victim_id]

        # L2 Distance Squared (hnswlib은 보통 제곱 유클리드 거리를 반환함)
        diff = vec_hub - vec_victim
        gt_dist_sq = np.dot(diff, diff)

        print(f"Ground Truth L2_sq({hub_id}, {victim_id}): {gt_dist_sq:.6f}")

        # 2-2. 강제 엣지 추가
        self.p.forced_insert_layer0_edge(hub_id, victim_id, False)

        # 2-3. HNSW 내부 계산값 확인
        # get_layer0_neighbors_with_distances()를 통해 내부에서 계산된 거리를 가져옴
        layout = self.p.get_layer0_neighbors_with_distances()
        neighbors = layout[hub_id] # (neighbor_id, distance) 튜플 리스트

        found_dist = None
        for n_id, dist in neighbors:
            if n_id == victim_id:
                found_dist = dist
                break

        self.assertIsNotNone(found_dist, "강제 추가된 엣지가 이웃 목록에 없습니다.")

        print(f"HNSW Internal Distance: {found_dist:.6f}")

        # 2-4. 비교 (허용 오차 범위 내 일치 여부)
        # float32 연산 오차 고려 (1e-5 정도)
        self.assertAlmostEqual(found_dist, gt_dist_sq, places=5,
                               msg="HNSW 내부 거리 계산값이 실제 수학적 거리와 다릅니다!")

        print("Distance check passed. (Internal calc matches Ground Truth)")

    def test_3_search_path_validation(self):
        """검증 3: 실제 검색 시 해당 엣지를 타고 가며, 그때 거리가 정상적인가?"""
        print("\n=== Test 3: Search Path & Distance Validation ===")

        # 시나리오: 0번 노드(Hub)와 99번 노드(Victim)를 연결하고
        # 99번을 찾을 때 0번을 거쳐가는지 확인
        hub = 0
        victim = 99

        self.p.forced_insert_layer0_edge(hub, victim, False)

        # 99번 노드(Victim) 자체를 쿼리로 검색
        query = self.data[victim]

        # Hub(0번)를 Entry Point로 강제하기 위해 ef=1로 설정하고
        # 내부적으로 0번이 시작점이 되길 기대하거나,
        # (확실하게 하려면 set_enter_point 같은 API가 필요하지만, 여기선 확률에 맡김)
        # 또는 0번 주변에서 검색 시작.

        # 여기서는 knn_query 결과에 victim이 포함되고, 그 거리가 0.0에 가까운지 확인
        labels, distances = self.p.knn_query(query, k=1)

        found_id = labels[0][0]
        found_dist = distances[0][0]

        print(f"Search Result -> ID: {found_id}, Dist: {found_dist}")

        self.assertEqual(found_id, victim, "Victim을 찾지 못했습니다.")
        self.assertAlmostEqual(found_dist, 0.0, places=5, msg="자기 자신과의 거리가 0이 아닙니다.")


    def test_4_search_path_validation(self):
                
        # Cosine space index
        dim = 5
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=10, ef_construction=50, M=4)

        # 두 벡터 추가 (정규화 안 된 상태)
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # 길이 1
        v2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # 길이 1, v1과 직교
        v3 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # v1과 45도 각도

        index.add_items(np.array([v1, v2, v3]), ids=np.array([0, 1, 2]))

        # 저장된 데이터 확인 (정규화되어 있어야 함)
        stored = index.get_items([0, 1, 2])
        print("Stored vectors (should be normalized):")
        print(f"v1: {stored[0]}, norm: {np.linalg.norm(stored[0]):.4f}")
        print(f"v2: {stored[1]}, norm: {np.linalg.norm(stored[1]):.4f}")
        print(f"v3: {stored[2]}, norm: {np.linalg.norm(stored[2]):.4f}")

        # Layer0 neighbors 확인
        neighbors = index.get_layer0_neighbors_with_distances()
        print("\nLayer0 neighbors with distances:")
        for node_id, neighbor_list in neighbors.items():
            print(f"Node {node_id}: {neighbor_list}")

        # 예상 거리 계산
        print("\n예상 거리 (cosine distance = 1 - cosine similarity):")
        print(f"v1 <-> v2: {1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)):.4f} (직교, ~1.0)")
        print(f"v1 <-> v3: {1.0 - np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)):.4f} (45도, ~0.293)")
        print(f"v2 <-> v3: {1.0 - np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)):.4f} (45도, ~0.293)")


if __name__ == '__main__':
    unittest.main()