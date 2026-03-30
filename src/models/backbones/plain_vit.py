import math
import torch
import torch.nn as nn
import timm

from aidall_seg.models.backbones import Backbone

class PlainViTBackbone(Backbone):
    """
    EoMT를 위한 단일 스케일(Plain) Vision Transformer 범용 래퍼 클래스.
    버전 종속성 없이 YAML을 통해 동적으로 모델을 로드.
    """
    def __init__(
        self,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        #논문 기반 수치들
        img_size: int = 640,
        patch_size: int = 16,
        l2_blocks: int = 4,
        pretrained: bool = True,
    ):
        # PyTorch 규약 준수: 모듈 할당 전 super().__init__ 먼저 호출 (임시값 전달)
        super().__init__(feat_channels=[1]) 
        
        self.backbone_name = backbone_name
        
        # timm 모델 로드 
        self.backbone = timm.create_model(
            backbone_name,
            img_size=img_size,
            patch_size=patch_size, 
            dynamic_img_size=True, 
            pretrained=pretrained,
            num_classes=0,  # 분류 헤드 제거 (출력 피처만 필요하기 때문)
        )
        
        # 모델 생성 후 실제 차원 값으로 feat_channels 덮어쓰기
        self.embed_dim = self.backbone.embed_dim
        self.feat_channels = [self.embed_dim]
        
        # 패치 사이즈 기반 업스케일 횟수 계산
        self.num_upscale = max(1, int(math.log2(patch_size)) - 2) #최소 1번은 업스케일하도록 보장
        
        # 이중 등록 방지: ModuleList로 감싸지 않고 인덱스만 저장
        total_blocks = len(self.backbone.blocks)
        self.l2_start = total_blocks - l2_blocks

    @property
    def l2_blocks(self):
        """
        메인 모델(eomt.py)에서 L2 블록을 직접 순회할 수 있도록 원본 참조를 반환.
        파라미터 이중 등록을 방지하는 안전한 노출 방식.
        """
        return self.backbone.blocks[self.l2_start:]

    @property
    def norm(self):
        """
        메인 모델에서 최종 LayerNorm을 적용할 수 있도록 노출.
        """
        return getattr(self.backbone, 'norm', None)

    def forward_l1(self, x: torch.Tensor) -> torch.Tensor:
        """
        이미지 패치화 및 L1 블록 통과
        """
        x = self.backbone.patch_embed(x)
        
        # timm 버전에 따른 위치 임베딩 (메서드 호출 vs 텐서 덧셈)
        if hasattr(self.backbone, '_pos_embed'):
            x = self.backbone._pos_embed(x)
        elif hasattr(self.backbone, 'pos_embed'):
            # callable이 아닌 텐서이므로 덧셈 연산 수행
            x = x + self.backbone.pos_embed 
            
        if hasattr(self.backbone, "norm_pre") and getattr(self.backbone, "norm_pre") is not None:
            x = self.backbone.norm_pre(x)
            
        # L1 블록 순회
        for blk in self.backbone.blocks[:self.l2_start]:
            x = blk(x)
            
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch 표준 호환성 및 단독 디버깅용 기본 forward 메서드.
        """
        x = self.forward_l1(x)
        
        for blk in self.l2_blocks:
            x = blk(x)
            
        if self.norm is not None:
            x = self.norm(x)
            
        return x
