import torch
from torchvision import transforms

# 画像前処理（MiniVLA の VisionEncoder に合わせる）
transform_pil = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# UCF101のクラス名を使ったより意味のあるプロンプト
UCF101_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
    "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo"
]


def collate_ucf101(batch, tokenizer):
    images = []
    texts = []
    labels = []

    for item in batch:
        img = item["images"]

        # 画像がすでに tensor の場合はそのまま使用
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[-1] in [1,3]:
                img = img.permute(2,0,1)  # (H,W,C)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224,224), mode="bilinear").squeeze(0)
        else:
            img = transform_pil(img)
        
        images.append(img)

        # UCF101データセットでは "actions" キーを使用
        label = item["actions"]
        labels.append(label)
        
        # 修正: クラス名を使った多様なプロンプト
        # 複数のプロンプトテンプレートをランダムに選択
        prompts = [
            f"Identify the action being performed.",
            f"What activity is shown in this image?",
            f"Classify this action.",
            f"What is the person doing?",
            f"Recognize the human activity."
        ]
        import random
        text = random.choice(prompts)
        texts.append(text)
        
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    return {
        "images": torch.stack(images),              # (B, 3, 224, 224)
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels),             # (B,)
    }
