import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FC_a(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC_a, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC_a(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FFN(nn.Module):
    def __init__(self, HIDDEN_SIZE,FF_SIZE,DROPOUT_R):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=FF_SIZE,
            out_size=HIDDEN_SIZE,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

class MHAtt(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(MHAtt, self).__init__()
        self.MULTI_HEAD = MULTI_HEAD
        self.HIDDEN_SIZE_HEAD = HIDDEN_SIZE_HEAD
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.DROPOUT_R = DROPOUT_R

        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted,map = self.att(v, k, q)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HIDDEN_SIZE
        )

        atted = atted.squeeze()
        atted = self.linear_merge(atted)

        return atted,map

    def att(self, value, key, query):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # att_map = F.softmax(scores, dim=-1)
        att_map = F.sigmoid(scores)
        map=att_map

        return torch.matmul(att_map, value),map

class MHAtt_9(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(MHAtt_9, self).__init__()
        self.MULTI_HEAD = MULTI_HEAD
        self.HIDDEN_SIZE_HEAD = HIDDEN_SIZE_HEAD
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.DROPOUT_R = DROPOUT_R

        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted,map = self.att(v, k, q)


        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HIDDEN_SIZE
        )

        atted = atted.squeeze()
        atted = self.linear_merge(atted)

        return atted,map

    def att(self, value, key, query):

        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        att_map = F.softmax(scores, dim=-1)
        map=att_map

        return torch.matmul(att_map, value),map

class GA(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(GA, self).__init__()

        self.mhatt1 = MHAtt(HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD)
        self.ffn = FFN(HIDDEN_SIZE,HIDDEN_SIZE*2,DROPOUT_R)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, y):
        x = x.squeeze()
        atted,map=self.mhatt1(y, y, x)
        x = self.norm1(x + self.dropout1(atted))
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x,map

class GA_9(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(GA_9, self).__init__()

        self.mhatt1 = MHAtt_9(HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD)
        self.ffn = FFN(HIDDEN_SIZE,HIDDEN_SIZE*2,DROPOUT_R)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, y):
        x = x.squeeze()
        atted,map=self.mhatt1(y, y, x)
        x = self.norm1(x + self.dropout1(atted))
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x,map

class MECOM(nn.Module):
    def __init__(self,drop_pro=0.1):
        super(MECOM, self).__init__()

        self.fc_2048_100 = nn.Linear(2048, 100)
        self.fc_4096_100 = nn.Linear(4096, 100)
        self.fc_2048_100_1 = nn.Linear(2048, 100)
        self.fc_2048_100_9 = nn.Linear(2048, 100)

        self.instance_image = nn.Sequential(
                nn.Linear(2048, 1),
                nn.ReLU()
        )
        self.instance_audio = nn.Sequential(
                nn.Linear(2048, 1),
                nn.ReLU()
        )
        self.instance_video = nn.Sequential(
                nn.Linear(2048, 1),
                nn.ReLU()
        )
        self.instance_text = nn.Sequential(
                nn.Linear(2048, 1),
                nn.ReLU()
        )

        if 1:
            self.image_audio = GA(2048, drop_pro, 8, 256)
            self.image_text = GA(2048, drop_pro, 8, 256)
            self.image_video = GA(2048, drop_pro, 8, 256)
            self.audio_image = GA(2048, drop_pro, 8, 256)
            self.audio_text = GA(2048, drop_pro, 8, 256)
            self.audio_video = GA(2048, drop_pro, 8, 256)
            self.text_image = GA(2048, drop_pro, 8, 256)
            self.text_audio = GA(2048, drop_pro, 8, 256)
            self.text_video = GA(2048, drop_pro, 8, 256)
            self.video_image = GA(2048, drop_pro, 8, 256)
            self.video_audio = GA(2048, drop_pro, 8, 256)
            self.video_text = GA(2048, drop_pro, 8, 256)

            self.image_audio_9 = GA_9(2048, drop_pro, 8, 256)
            self.image_text_9 = GA_9(2048, drop_pro, 8, 256)
            self.image_video_9 = GA_9(2048, drop_pro, 8, 256)
            self.audio_image_9 = GA_9(2048, drop_pro, 8, 256)
            self.audio_text_9 = GA_9(2048, drop_pro, 8, 256)
            self.audio_video_9 = GA_9(2048, drop_pro, 8, 256)
            self.text_image_9 = GA_9(2048, drop_pro, 8, 256)
            self.text_audio_9 = GA_9(2048, drop_pro, 8, 256)
            self.text_video_9 = GA_9(2048, drop_pro, 8, 256)
            self.video_image_9 = GA_9(2048, drop_pro, 8, 256)
            self.video_audio_9 = GA_9(2048, drop_pro, 8, 256)
            self.video_text_9 = GA_9(2048, drop_pro, 8, 256)

        if 1:
            self.de_image_2048 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

            self.de_audio_2048 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

            self.de_text_2048 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

            self.de_video_2048 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )
            self.de_image_2048_9 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

            self.de_audio_2048_9 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

            self.de_text_2048_9 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

            self.de_video_2048_9 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
            )

    def Loss_re1(self,res,gt,sn):
        return torch.sum(torch.sum(torch.pow(res-gt,2),dim=1) * sn)

    def forward(self, image, audio, text, video, image_9, audio_9, text_9, video_9, mask, mask_re):

        image_audio_a,map = self.image_audio(image, audio)
        image_text_a,map = self.image_text(image, text)
        image_video_a,map = self.image_video(image, video)
        image_a = (image_audio_a + image_text_a + image_video_a) / 3
        image_all = torch.mul(image_a, mask[0].reshape(mask.size()[1], 1))

        audio_image_a,map = self.audio_image(audio, image)
        audio_text_a,map = self.audio_text(audio, text)
        audio_video_a,map = self.audio_video(audio, video)
        audio_a = (audio_image_a + audio_text_a + audio_video_a) / 3
        audio_all = torch.mul(audio_a, mask[1].reshape(mask.size()[1], 1))

        text_image_a,map = self.text_image(text, image)
        text_audio_a,map = self.text_audio(text, audio)
        text_video_a,map = self.text_video(text, video)
        text_a = (text_image_a + text_audio_a + text_video_a) / 3
        text_all = torch.mul(text_a, mask[2].reshape(mask.size()[1], 1))

        video_image_a,map = self.video_image(video, image)
        video_audio_a,map = self.video_audio(video, audio)
        video_text_a,map = self.video_text(video, text)
        video_a = (video_image_a + video_audio_a + video_text_a) / 3
        video_all = torch.mul(video_a, mask[3].reshape(mask.size()[1], 1))

        x = image_all + audio_all + text_all + video_all

        deco_image = self.de_image_2048(x)
        deco_audio = self.de_audio_2048(x)
        deco_text = self.de_text_2048(x)
        deco_video = self.de_video_2048(x)

        full_image = deco_image * mask_re[0].reshape(mask_re.size()[1], 1) + image_all
        full_audio = deco_audio * mask_re[1].reshape(mask_re.size()[1], 1) + audio_all
        full_text = deco_text * mask_re[2].reshape(mask_re.size()[1], 1) + text_all
        full_video = deco_video * mask_re[3].reshape(mask_re.size()[1], 1) + video_all

        full_x = full_image + full_audio + full_text + full_video

        #_9 start

        # Gumbel_Softmax
        score_image = self.instance_image(image_9)
        score_image = score_image.squeeze()
        score_hard_image = F.gumbel_softmax(score_image, dim=-1, hard=True)
        score_hard_image = score_hard_image.unsqueeze(dim=-1)
        image_9 = score_hard_image * image_9

        score_audio = self.instance_audio(audio_9)
        score_audio = score_audio.squeeze()
        score_hard_audio = F.gumbel_softmax(score_audio, dim=-1, hard=True)
        score_hard_audio = score_hard_audio.unsqueeze(dim=-1)
        audio_9 = score_hard_audio * audio_9

        score_text = self.instance_text(text_9)
        score_text = score_text.squeeze()
        score_hard_text = F.gumbel_softmax(score_text, dim=-1, hard=True)
        score_hard_text = score_hard_text.unsqueeze(dim=-1)
        text_9 = score_hard_text * text_9

        score_video = self.instance_video(video_9)
        score_video = score_video.squeeze()
        score_hard_video = F.gumbel_softmax(score_video, dim=-1, hard=True)
        score_hard_video = score_hard_video.unsqueeze(dim=-1)
        video_9 = score_hard_video * video_9


        image_audio_a_9, map = self.image_audio_9(image_9, audio_9)
        image_text_a_9, map = self.image_text_9(image_9, text_9)
        image_video_a_9, map_i_v = self.image_video_9(image_9, video_9)
        image_a_9 = (image_audio_a_9 + image_text_a_9 + image_video_a_9) / 3
        image_all_9 = torch.mul(image_a_9, mask[0].unsqueeze(1).unsqueeze(1))

        audio_image_a_9, map = self.audio_image_9(audio_9, image_9)
        audio_text_a_9, map = self.audio_text_9(audio_9, text_9)
        audio_video_a_9, map = self.audio_video_9(audio_9, video_9)
        audio_a_9 = (audio_image_a_9 + audio_text_a_9 + audio_video_a_9) / 3
        audio_all_9 = torch.mul(audio_a_9, mask[1].unsqueeze(1).unsqueeze(1))

        text_image_a_9, map = self.text_image_9(text_9, image_9)
        text_audio_a_9, map = self.text_audio_9(text_9, audio_9)
        text_video_a_9, map = self.text_video_9(text_9, video_9)
        text_a_9 = (text_image_a_9 + text_audio_a_9 + text_video_a_9) / 3
        text_all_9 = torch.mul(text_a_9, mask[2].unsqueeze(1).unsqueeze(1))

        video_image_a_9, map_v_i = self.video_image_9(video_9, image_9)
        video_audio_a_9, map = self.video_audio_9(video_9, audio_9)
        video_text_a_9, map = self.video_text_9(video_9, text_9)
        video_a_9 = (video_image_a_9 + video_audio_a_9 + video_text_a_9) / 3
        video_all_9 = torch.mul(video_a_9, mask[3].unsqueeze(1).unsqueeze(1))

        image_all_9 = torch.sum(image_all_9, dim=1) / 2
        audio_all_9 = torch.sum(audio_all_9, dim=1) / 2
        text_all_9 = torch.sum(text_all_9, dim=1) / 2
        video_all_9 = torch.sum(video_all_9, dim=1) / 2
        x_9 = image_all_9 + audio_all_9 + text_all_9 + video_all_9

        deco_image_9 = self.de_image_2048_9(x_9)
        deco_audio_9 = self.de_audio_2048_9(x_9)
        deco_text_9 = self.de_text_2048_9(x_9)
        deco_video_9 = self.de_video_2048_9(x_9)

        full_image_9 = deco_image_9 * mask_re[0].reshape(mask_re.size()[1], 1) + image_all_9
        full_audio_9 = deco_audio_9 * mask_re[1].reshape(mask_re.size()[1], 1) + audio_all_9
        full_text_9 = deco_text_9 * mask_re[2].reshape(mask_re.size()[1], 1) + text_all_9
        full_video_9 = deco_video_9 * mask_re[3].reshape(mask_re.size()[1], 1) + video_all_9

        full_x_9 = full_image_9 + full_audio_9 + full_text_9 + full_video_9

        final_x = torch.cat([full_x,full_x_9],dim=1)

        logits = self.fc_4096_100(final_x)
        logits_1 = self.fc_2048_100_1(full_x)
        logits_9 = self.fc_2048_100_9(full_x_9)

        return logits, image_all, audio_all, text_all, video_all, deco_image, deco_audio, deco_text, deco_video, x, image_all_9, audio_all_9, text_all_9, video_all_9, \
            deco_image_9, deco_audio_9, deco_text_9, deco_video_9, x_9,logits_1,logits_9,final_x
