# Chapter 2. 그래프 모형과 응용

**규칙 1 (사슬(chain)에서의 조건부 독립)**
$X$와 $Y$ 사이에 하나의 방향성 경로 (unidirectional path)만 있고 $Z$가 해당 경로를 가로막는 변수 집합인 경우, $Z$가 조건부로 주어졌을 때 두 변수 $X$와 $Y$는 조건부 독립이다.

**규칙 2 (분기(fork)에서의 조건부 독립)**
변수 $X$가 변수 $Y$와 변수 $Z$의 공통 원인 (common cause) 이고 $Y$와 $Z$사이에 단 하나의 경로가 있는 경우 $X$의 조건이 주어졌을 때 $Y$와 $Z$는 조건부 독립이다.

**규칙 3 (충돌부(collider)에서의 조건부 독립)**
변수 $Z$가 두 변수 $X$와 $Y$ 사이의 충돌 노드이고 $X$와 $Y$ 사이에 단 하나의 경로 (one path) 만 있는 경우, $X$와 $Y$는 비조건부 독립 (unconditionally independent)이다. 그러나 $Z$ 또는 $Z$의 자손을 조건부로 하였을 때에는 종속적일 가능성이 있다.

**정의 2.4.1 (d-seperated)**
경로 $p$는 노드 $Z$에 의해 차단된다는 명제는 다음 항목과 필요충분조건이다.
1. 경로 $p$는 중간노드 $B$가 $Z$에 속하는 (즉, $B$가 조건부로 설정됨) 사슬 $A \rightarrow B \rightarrow C$  또는 분기 $A \leftarrow B \rightarrow C$를 포함한다.(조건부 chain, fork)
2. 경로 $p$는 중간노드 $B$와 $B$의 자손이 $Z$에 속하지 않는 충돌부 $A \rightarrow B \leftarrow C$를 포함한다.(비조건부 colliders)

