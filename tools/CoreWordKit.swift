// CoreWordKit.swift
// 共享的核心词提取库：N-gram + NPMI + 左右信息熵 + 停用词 / 长词优先。
// 由 ExtractCoreWords.swift（CLI）与 ClusterBooks.swift（批量聚类）共同编译引用。
// 纯 Foundation，零第三方依赖。
//
// 内存设计：汉字映射 UInt16 ID 压成单条流，N-gram 打包成 16 字节 GramKey；
// 左右邻居用扁平计数 + Σc·log2c 累加求熵，避免每个候选一个小字典；
// 熵阈值 >0 时跳过频次 1 的 gram（该侧熵必为 0），结果不变但大幅提速。
// ---------------------------------------------------------------------------

import Foundation
#if canImport(CoreFoundation)
import CoreFoundation
#endif

let textSuffixes: Set<String> = ["txt", "md", "markdown", "text"]
let sentinelID: UInt16 = 0      // 片段分隔符 / BOS / EOS 共用的哨兵

// MARK: - 数据模型

struct CoreWord: Codable, Equatable {
    let word: String
    let freq: Int
    let pmi: Double
    let npmi: Double
    let leftEntropy: Double
    let rightEntropy: Double

    enum CodingKeys: String, CodingKey {
        case word, freq, pmi, npmi
        case leftEntropy = "left_entropy"
        case rightEntropy = "right_entropy"
    }
}

struct ExtractOptions: Codable {
    var top: Int = 500
    var ngramMin: Int = 2
    var ngramMax: Int = 5
    var minNPMI: Double = 0.3
    var minLeftEntropy: Double = 1.0
    var minRightEntropy: Double = 1.0
    var minFreq: Int = 1
    var boundary: BoundaryMode = .count
    var keepASCII: Bool = false
    var maxChars: Int = 0

    // 新增：文本过滤与长词优先
    var useStopwords: Bool = true
    var useBoundaryFilter: Bool = true
    var useSubstringFilter: Bool = true
    var substringRatio: Double = 0.5
    var stopwordsPath: String = ""
    var lengthBonus: Double = 0

    enum BoundaryMode: String, Codable {
        case count
        case ignore
    }

    enum CodingKeys: String, CodingKey {
        case top
        case ngramMin = "ngram_min"
        case ngramMax = "ngram_max"
        case minNPMI = "min_npmi"
        case minLeftEntropy = "min_left_entropy"
        case minRightEntropy = "min_right_entropy"
        case minFreq = "min_freq"
        case boundary, keepASCII = "keep_ascii", maxChars = "max_chars"
        case useStopwords = "use_stopwords"
        case useBoundaryFilter = "use_boundary_filter"
        case useSubstringFilter = "use_substring_filter"
        case substringRatio = "substring_ratio"
        case stopwordsPath = "stopwords_path"
        case lengthBonus = "length_bonus"
    }
}

struct ExtractionResult: Codable {
    let words: [CoreWord]
    let stats: Stats
    let params: ExtractOptions

    struct Stats: Codable {
        let totalChars: Int
        let totalSegments: Int
        let ngramTypes: Int
        let npmiSurvivors: Int
        let entropySurvivors: Int
        let stopwordRemoved: Int
        let boundaryRemoved: Int
        let substringRemoved: Int
        let returned: Int

        enum CodingKeys: String, CodingKey {
            case totalChars = "total_chars"
            case totalSegments = "total_segments"
            case ngramTypes = "ngram_types"
            case npmiSurvivors = "npmi_survivors"
            case entropySurvivors = "entropy_survivors"
            case stopwordRemoved = "stopword_removed"
            case boundaryRemoved = "boundary_removed"
            case substringRemoved = "substring_removed"
            case returned
        }
    }
}

// MARK: - Gram 键与字符编码

/// 16 字节紧凑键：a 存前 4 个 UInt16 字符 ID，b 低 16 位存第 5 个、高位存长度。
struct GramKey: Hashable {
    let a: UInt64
    let b: UInt32

    var length: Int { Int((b >> 16) & 0x7) }

    @inline(__always)
    func ids() -> [UInt16] {
        let n = length
        var out: [UInt16] = []
        out.reserveCapacity(n)
        for k in 0..<min(n, 4) {
            out.append(UInt16((a >> UInt64(16 * k)) & 0xFFFF))
        }
        if n == 5 { out.append(UInt16(b & 0xFFFF)) }
        return out
    }

    @inline(__always)
    static func make(_ stream: [UInt16], _ i: Int, _ n: Int) -> GramKey {
        var a: UInt64 = 0
        for k in 0..<min(n, 4) {
            a |= UInt64(stream[i + k]) << UInt64(16 * k)
        }
        var b = UInt32(n) << 16
        if n == 5 { b |= UInt32(stream[i + 4]) }
        return GramKey(a: a, b: b)
    }

    @inline(__always)
    static func make(_ ids: [UInt16]) -> GramKey {
        var a: UInt64 = 0
        for k in 0..<min(ids.count, 4) {
            a |= UInt64(ids[k]) << UInt64(16 * k)
        }
        var b = UInt32(ids.count) << 16
        if ids.count == 5 { b |= UInt32(ids[4]) }
        return GramKey(a: a, b: b)
    }
}

/// 汉字 ⇄ UInt16 ID 双向表（ID 0 保留给哨兵）。
struct GramCodec {
    private(set) var chars: [Character] = ["\u{0}"]
    private var ids: [Character: UInt16] = [:]

    mutating func id(for ch: Character) -> UInt16 {
        if let existing = ids[ch] { return existing }
        guard chars.count < Int(UInt16.max) else { return sentinelID }
        let newID = UInt16(chars.count)
        ids[ch] = newID
        chars.append(ch)
        return newID
    }

    func character(_ id: UInt16) -> Character {
        Int(id) < chars.count ? chars[Int(id)] : "\u{0}"
    }

    func key(_ word: String) -> GramKey {
        GramKey.make(word.map { ids[$0] ?? sentinelID })
    }

    func word(_ key: GramKey) -> String {
        String(key.ids().map { character($0) })
    }
}

// MARK: - 停用词与碎片规则

enum WordFilters {
    /// 项目早期精选的中文功能词（保留合并，避免丢失「实际上 / 的时候 / 之内」等针对性词）。
    static let legacyBuiltinStopwords: Set<String> = Set("""
我们 你们 他们 它们 咱们 自己 这个 那个 这些 那些 这种 那种 什么 怎么 怎样 为何 哪里
因为 所以 但是 然而 虽然 如果 假如 即使 尽管 而且 并且 或者 还是 以及 于是 因此 由于
关于 对于 通过 根据 按照 为了 除了 由于 关于 并且 况且 何况 与其 不如 不过 只是
可以 能够 应该 必须 需要 可能 似乎 其实 基本 主要 一般 一定 非常 十分 相当 比较
更加 逐渐 一直 已经 正在 曾经 仍然 依然 忽然 突然 显然 当然 自然 确实 实际上
进行 成为 作为 出现 存在 发生 产生 具有 包括 使用 采用 提供 实现 表示 认为 发现
一个 一种 一些 一次 一下 一点 一时 第一 第二 第三 这样 那样 如此 其中 之后 之前
以后 以前 现在 当时 同时 目前 开始 左右 上下 方面 情况 问题 时候 时间 地方 部分
没有 不是 就是 还有 也是 都是 一种 是一样的 为主 可见 看出 体现 甚至 改变 更多
先生 太太 小姐 地说 地说 起来 一样 到了 到了这种 到了那个 知道 我们 他们 什么
这一 那一 之一 之一 之中 之间 之内 之外 一般 一切 一切 所有 任何 每个 各个
这是一种 是一种 就可以了 就可以了 的时候 就是说 也就是说 事实上 实际上 总的 来说
""".split(whereSeparator: \.isWhitespace).map(String.init))

    /// 汇总的中英文停用词表，编译期嵌入（数据文件 tools/stopwords_full.txt，共 2992 条）。
    /// 由脚本从该数据文件生成；后续增删请改数据文件后重新生成此段，勿手工编辑。
    static let stopwordsFullRaw: String = """
"
#
$
&
'
'll
've
(
)
*
+
,
-
--
.
...
......
...................
./
.一
.数
.日
/
//
0
0o
0s
1
2
3
3a
3b
3d
4
5
6
6b
6o
7
8
9
:
://
::
;
<
=
>
?
@
I
Lex
ZT
ZZ
[
]
_
a
a's
a1
a2
a3
a4
ab
able
about
above
abroad
abst
ac
accordance
according
accordingly
across
act
actually
ad
added
adj
adopted
ae
af
affected
affecting
affects
after
afterwards
ag
again
against
ago
ah
ahead
ain
ain't
aj
al
all
allow
allows
almost
alone
along
alongside
already
also
although
always
am
amid
amidst
among
amongst
amoungst
amount
an
and
announce
another
any
anybody
anyhow
anymore
anyone
anything
anyway
anyways
anywhere
ao
ap
apart
apparently
appear
appreciate
appropriate
approximately
ar
are
aren
aren't
arent
arise
around
as
aside
ask
asking
associated
at
au
auth
av
available
aw
away
awfully
ax
ay
az
b
b1
b2
b3
ba
back
backward
backwards
bc
bd
be
became
because
become
becomes
becoming
been
before
beforehand
begin
beginning
beginnings
begins
behind
being
believe
below
beside
besides
best
better
between
beyond
bi
bill
biol
bj
bk
bl
bn
both
bottom
bp
br
brief
briefly
bs
bt
bu
but
bx
by
c
c'mon
c's
c1
c2
c3
ca
call
came
can
can't
cannot
cant
caption
cause
causes
cc
cd
ce
certain
certainly
cf
cg
ch
changes
ci
cit
cj
cl
clearly
cm
cn
co
co.
com
come
comes
computer
con
concerning
consequently
consider
considering
contain
containing
contains
corresponding
could
couldn
couldn't
couldnt
course
cp
cq
cr
cry
cs
ct
cu
currently
cv
cx
cy
cz
d
d2
da
dare
daren't
date
dc
dd
de
definitely
describe
described
despite
detail
df
di
did
didn
didn't
different
directly
dj
dk
dl
do
does
doesn
doesn't
doing
don
don't
done
down
downwards
dp
dr
ds
dt
du
due
during
dx
dy
e
e2
e3
ea
each
ec
ed
edu
ee
ef
effect
eg
ei
eight
eighty
either
ej
el
eleven
else
elsewhere
em
empty
en
end
ending
enough
entirely
eo
ep
eq
er
es
especially
est
et
et-al
etc
eu
ev
even
ever
evermore
every
everybody
everyone
everything
everywhere
ex
exactly
example
except
exp
ey
f
f2
fa
fairly
far
farther
fc
few
fewer
ff
fi
fifteen
fifth
fifty
fify
fill
find
fire
first
five
fix
fj
fl
fn
fo
followed
following
follows
for
forever
former
formerly
forth
forty
forward
found
four
fr
from
front
fs
ft
fu
full
further
furthermore
fy
g
ga
gave
ge
get
gets
getting
gi
give
given
gives
giving
gj
gl
go
goes
going
gone
got
gotten
gr
greetings
gs
gy
h
h2
h3
had
hadn
hadn't
half
happens
hardly
has
hasn
hasn't
hasnt
have
haven
haven't
having
he
he'd
he'll
he's
hed
hello
help
hence
her
here
here's
hereafter
hereby
herein
heres
hereupon
hers
herse
herself
hes
hh
hi
hid
him
himse
himself
his
hither
hj
ho
home
hopefully
how
how's
howbeit
however
hr
hs
http
hu
hundred
hy
i
i'd
i'll
i'm
i've
i2
i3
i4
i6
i7
i8
ia
ib
ibid
ic
id
ie
if
ig
ignored
ih
ii
ij
il
im
immediate
immediately
importance
important
in
inasmuch
inc
inc.
indeed
index
indicate
indicated
indicates
information
inner
inside
insofar
instead
interest
into
invention
inward
io
ip
iq
ir
is
isn
isn't
it
it'd
it'll
it's
itd
its
itself
itse”
iv
ix
iy
iz
j
jj
jr
js
jt
ju
just
k
ke
keep
keeps
kept
keys
kg
kj
km
know
known
knows
ko
l
l2
la
largely
last
lately
later
latter
latterly
lb
lc
le
least
les
less
lest
let
let's
lets
lf
like
liked
likely
likewise
line
little
lj
ll
ln
lo
look
looking
looks
los
low
lower
lr
ls
lt
ltd
m
m2
ma
made
mainly
make
makes
many
may
maybe
mayn't
me
mean
means
meantime
meanwhile
merely
mg
might
mightn
mightn't
mill
million
mine
minus
miss
ml
mn
mo
more
moreover
most
mostly
move
mr
mrs
ms
mt
mu
much
mug
must
mustn
mustn't
my
myself
myse”
n
n2
na
name
namely
nay
nc
nd
ne
near
nearly
necessarily
necessary
need
needn
needn't
needs
neither
never
neverf
neverless
nevertheless
new
next
ng
ni
nine
ninety
nj
nl
nn
no
no-one
nobody
non
none
nonetheless
noone
nor
normally
nos
not
noted
nothing
notwithstanding
novel
now
nowhere
nr
ns
nt
ny
o
oa
ob
obtain
obtained
obviously
oc
od
of
off
often
og
oh
oi
oj
ok
okay
ol
old
om
omitted
on
once
one
one's
ones
only
onto
oo
op
opposite
oq
or
ord
os
ot
other
others
otherwise
ou
ought
oughtn't
our
ours
ourselves
out
outside
over
overall
ow
owing
own
ox
oz
p
p1
p2
p3
page
pagecount
pages
par
part
particular
particularly
pas
past
pc
pd
pe
per
perhaps
pf
ph
pi
pj
pk
pl
placed
please
plus
pm
pn
po
poorly
possible
possibly
potentially
pp
pq
pr
predominantly
present
presumably
previously
primarily
probably
promptly
proud
provided
provides
ps
pt
pu
put
py
q
qj
qu
que
quickly
quite
qv
r
r2
ra
ran
rather
rc
rd
re
readily
really
reasonably
recent
recently
ref
refs
regarding
regardless
regards
related
relatively
research
research-articl
respectively
resulted
resulting
results
rf
rh
ri
right
rj
rl
rm
rn
ro
round
rq
rr
rs
rt
ru
run
rv
ry
s
s2
sa
said
same
saw
say
saying
says
sc
sd
se
sec
second
secondly
section
see
seeing
seem
seemed
seeming
seems
seen
self
selves
sensible
sent
serious
seriously
seven
several
sf
shall
shan
shan't
she
she'd
she'll
she's
shed
shes
should
should've
shouldn
shouldn't
show
showed
shown
showns
shows
si
side
significant
significantly
similar
similarly
since
sincere
six
sixty
sj
sl
slightly
sm
sn
so
some
somebody
someday
somehow
someone
somethan
something
sometime
sometimes
somewhat
somewhere
soon
sorry
sp
specifically
specified
specify
specifying
sq
sr
ss
st
state
states
still
stop
strongly
sub
substantially
successfully
such
sufficiently
suggest
sup
sure
sy
system
sz
t
t's
t1
t2
t3
take
taken
taking
tb
tc
td
te
tell
ten
tends
tf
th
than
thank
thanks
thanx
that
that'll
that's
that've
thats
the
their
theirs
them
themselves
then
thence
there
there'd
there'll
there're
there's
there've
thereafter
thereby
thered
therefore
therein
thereof
therere
theres
thereto
thereupon
these
they
they'd
they'll
they're
they've
theyd
theyre
thick
thickv
thin
thing
things
think
third
thirty
this
thorough
thoroughly
those
thou
though
thoughh
thousand
three
throug
through
throughout
thru
thus
ti
til
till
tip
tj
tl
tm
tn
to
together
too
took
top
toward
towards
tp
tq
tr
tried
tries
truly
try
trying
ts
tt
tv
twelve
twenty
twice
two
tx
u
u201d
ue
ui
uj
uk
um
un
under
underneath
undoing
unfortunately
unless
unlike
unlikely
until
unto
uo
up
upon
ups
upwards
ur
us
use
used
useful
usefully
usefulness
uses
using
usually
ut
uucp
v
va
value
various
vd
ve
versus
very
via
viz
vj
vo
vol
vols
volumtype
vq
vs
vt
vu
w
wa
want
wants
was
wasn
wasn't
wasnt
way
we
we'd
we'll
we're
we've
wed
welcome
well
well-b
went
were
weren
weren't
werent
what
what'll
what's
what've
whatever
whats
when
when's
whence
whenever
where
where's
whereafter
whereas
whereby
wherein
wheres
whereupon
wherever
whether
which
whichever
while
whilst
whim
whither
who
who'd
who'll
who's
whod
whoever
whole
whom
whomever
whos
whose
why
why's
wi
widely
will
willing
wish
with
within
without
wo
won
won't
wonder
wont
words
world
would
wouldn
wouldn't
wouldnt
www
x
x1
x2
x3
xf
xi
xj
xk
xl
xn
xo
xs
xt
xv
xx
y
y2
yes
yet
yj
yl
you
you'd
you'll
you're
you've
youd
your
youre
yours
yourself
yourselves
yr
ys
yt
z
zero
zi
zt
zz
}
~~~~
·
×
×××
Δ
Ψ
γ
μ
φ
φ．
В
—
——
———
‘
’
’‘
“
”
”，
……
…………………………………………………③
′∈
′｜
℃
Ⅲ
↑
→
∈［
∪φ∈
≈
①
②
②ｃ
③
③］
④
⑤
⑥
⑦
⑧
⑨
⑩
──
■
▲
、
。
〉
《
》
》），
」
『
』
【
】
〔
〕
〕〔
㈧
一
一.
一一
一下
一个
一些
一何
一切
一则
一则通过
一天
一定
一方面
一旦
一时
一来
一样
一次
一片
一番
一直
一致
一般
一起
一转眼
一边
一面
万一
三天两头
三番两次
三番五次
上
上下
上升
上去
上来
上述
上面
下
下列
下去
下来
下面
不
不一
不下
不久
不了
不亦乐乎
不仅
不仅仅
不仅仅是
不会
不但
不光
不免
不再
不力
不单
不变
不只
不可
不可开交
不可抗拒
不同
不外
不外乎
不够
不大
不如
不妨
不定
不对
不少
不尽
不尽然
不巧
不已
不常
不得
不得不
不得了
不得已
不必
不怎么
不怕
不惟
不成
不拘
不择手段
不敢
不料
不断
不日
不时
不是
不曾
不止
不止一次
不比
不消
不满
不然
不然的话
不特
不独
不由得
不知不觉
不管
不管怎样
不经意
不胜
不能
不能不
不至于
不若
不要
不论
不起
不足
不过
不迭
不问
不限
与
与其
与其说
与否
与此同时
专门
且
且不说
且说
两者
严格
严重
个
个人
个别
中小
中间
丰富
临
临到
为
为主
为了
为什么
为什麽
为何
为止
为此
为着
主张
主要
举凡
举行
乃
乃至
乃至于
么
之
之一
之前
之后
之後
之所以
之类
乌乎
乎
乘
乘势
乘机
乘虚
乘隙
也
也好
也就是说
也是
也罢
了
了解
争取
二来
二话不说
二话没说
于
于是
于是乎
云云
云尔
互相
些
交口
亦
产生
亲口
亲手
亲眼
亲自
亲身
人
人人
人们
人家
什么
什么样
什麽
今
今后
今天
今年
今後
介于
仍
仍旧
仍然
从
从不
从严
从中
从事
从今以后
从优
从古到今
从古至今
从头
从宽
从小
从新
从无到有
从早到晚
从未
从来
从此
从此以后
从而
从轻
从速
从重
他
他人
他们
他是
他的
代替
以
以上
以下
以为
以便
以免
以前
以及
以后
以外
以後
以故
以期
以来
以至
以至于
以致
们
任
任何
任凭
任务
企图
伙同
会
伟大
传说
传闻
似乎
似的
但
但凡
但愿
但是
何
何乐而不为
何以
何况
何处
何妨
何尝
何必
何时
何止
何苦
何须
余外
作为
你
你们
你是
你的
使
使得
使用
例如
依
依据
依照
依靠
便于
促进
保持
俺
俺们
倍加
倍感
倒不如
倒不如说
倒是
倘
倘使
倘或
倘然
倘若
借
借以
借此
假使
假如
假若
偏偏
做到
偶尔
偶而
傥然
像
儿
允许
元／吨
充其极
充其量
充分
先不先
先后
先後
先生
光是
全体
全力
全年
全然
全身心
全部
全都
全面
八成
公然
兮
共同
共总
关于
其
其一
其中
其二
其他
其余
其后
其它
其实
其次
具体
具体地说
具体来说
具体说来
具有
兼之
内
再
再其次
再则
再有
再次
再者
再者说
再说
冒
冲
决不
决定
决非
况且
准备
凑巧
凝神
几
几乎
几度
几时
几番
几经
凡
凡是
凭
凭借
出于
出去
出来
出现
分别
分头
分期分批
切不可
切切
切勿
切莫
则
则甚
刚好
刚巧
刚才
别
别人
别处
别是
别的
别管
别说
到
到了儿
到处
到头
到头来
到底
到目前为止
前后
前此
前者
前进
前面
加上
加之
加以
加入
加强
动不动
动辄
勃然
匆匆
十分
千万千万
单单
单纯
即
即令
即使
即便
即刻
即如
即将
即或
即是说
即若
却
却不
原来
去
又
又及
及
及其
及时
及至
双方
反之
反之亦然
反之则
反倒
反倒是
反应
反手
反映
反而
反过来
反过来说
取得
取道
受到
变成
另
另一个
另一方面
另外
另悉
另方面
另行
只
只当
只怕
只是
只有
只消
只要
只限
叫
叫做
召开
叮咚
可
可以
可好
可是
可能
可见
各
各个
各人
各位
各地
各式
各种
各级
各自
合理
同
同一
同时
同样
后
后来
后者
后面
向
向使
向着
吓
吗
否则
吧
吧哒
吱
呀
呃
呆呆地
呕
呗
呜
呜呼
呢
周围
呵
呵呵
呸
呼哧
呼啦
咋
和
咚
咦
咧
咱
咱们
咳
哇
哈
哈哈
哉
哎
哎呀
哎哟
哗
哗啦
哟
哦
哩
哪
哪个
哪些
哪儿
哪天
哪年
哪怕
哪样
哪边
哪里
哼
哼唷
唉
唯有
啊
啊呀
啊哈
啊哟
啐
啥
啦
啪达
啷当
喂
喏
喔唷
喽
嗡
嗡嗡
嗬
嗯
嗳
嘎
嘎嘎
嘎登
嘘
嘛
嘻
嘿
嘿嘿
因
因为
因了
因此
因着
因而
固然
在
在下
在于
地
坚决
坚持
基于
基本
基本上
处在
处处
处理
复杂
多
多么
多亏
多多
多多少少
多多益善
多少
多年前
多年来
多数
多次
够瞧的
大
大不了
大举
大体上
大凡
大力
大多
大多数
大大
大家
大张旗鼓
大批
大抵
大概
大略
大约
大致
大都
大量
大面儿上
失去
奋勇
她
她们
她是
她的
好
好在
好的
好象
如
如上
如上所述
如下
如今
如何
如其
如前所述
如同
如常
如是
如期
如果
如次
如此
如此等等
如若
始而
姑且
存在
存心
孰料
孰知
宁
宁可
宁愿
宁肯
它
它们
它们的
它是
它的
安全
完全
完成
实现
实际
宣布
容易
密切
对
对于
对应
对待
对方
对比
将
将才
将要
将近
小
少数
尔
尔后
尔尔
尔等
尚且
尤其
就
就地
就是
就是了
就是说
就此
就算
就要
尽
尽可能
尽如人意
尽心尽力
尽心竭力
尽快
尽早
尽然
尽管
尽管如此
尽量
局外
居然
届时
属于
屡屡
屡次
屡次三番
岂但
岂止
岂非
川流不息
左右
巨大
巩固
差一点
差不多
己
已
已矣
已经
巴
巴巴
帮助
常常
常言说
常言说得好
常言道
平素
年复一年
并
并不
并不是
并且
并排
并无
并没
并没有
并肩
并非
广大
广泛
应当
应用
应该
庶乎
庶几
开外
开始
开展
引起
弹指之间
强烈
强调
归
归根到底
归根结底
归齐
当
当下
当中
当儿
当前
当即
当口儿
当地
当场
当头
当庭
当时
当然
当真
当着
形成
彻夜
彻底
彼
彼时
彼此
往
往往
待
待到
很
很多
很少
後来
後面
得
得了
得出
得到
得天独厚
得起
心里
必定
必将
必然
必要
必须
快要
忽地
忽然
怎
怎么
怎么办
怎么样
怎奈
怎样
怎麽
急匆匆
怪不得
总之
总是
总的来看
总的来说
总的说来
总结
总而言之
恍然
恐怕
恰似
恰好
恰如
恰巧
恰恰
恰恰相反
恰逢
您
您们
您是
惟其
惯常
意思
愤然
愿意
慢说
成为
成年累月
成心
我
我们
我是
我的
或
或则
或多或少
或是
或曰
或者
或许
战斗
截然
截至
所
所以
所在
所幸
所有
所谓
才
才能
扑通
打
打从
打开天窗说亮话
扩大
把
抑或
抽冷子
拦腰
拿
按
按时
按期
按照
按理
按说
挨个
挨家挨户
挨次
挨着
挨门挨户
挨门逐户
换句话说
换言之
据
据实
据悉
据我所知
据此
据称
据说
掌握
接下来
接着
接著
接连不断
故
故意
故此
故而
敞开儿
敢于
敢情
数/
整个
断然
方便
方才
方能
方面
旁人
无
无宁
无法
无论
既
既往
既是
既然
日复一日
日渐
日益
日臻
日见
时候
昂然
明显
明确
是
是不是
是以
是否
是的
显然
显著
普通
普遍
暗中
暗地里
暗自
更为
更加
更进一步
曾
曾经
替
替代
最
最后
最大
最好
最後
最近
最高
有
有些
有关
有利
有力
有及
有所
有效
有时
有点
有的
有的是
有着
有著
望
朝
朝着
末##末
本
本人
本地
本着
本身
权时
来
来不及
来得及
来看
来着
来自
来讲
来说
极为
极了
极其
极力
极大
极度
极端
构成
果然
果真
某
某个
某些
某某
根据
根本
格外
次第
欢迎
欤
正值
正在
正如
正巧
正常
正是
此
此中
此后
此地
此处
此外
此时
此次
此间
毋宁
每
每个
每天
每年
每当
每时每刻
每每
每逢
比
比及
比如
比如说
比方
比照
比起
比较
毕竟
毫不
毫无
毫无例外
毫无保留地
沙沙
没奈何
没有
沿
沿着
注意
深入
清楚
满足
漫说
焉
然则
然后
然後
然而
照
照着
牢牢
特别是
特殊
特点
犹且
犹自
独自
猛然
猛然间
率尔
率然
现代
现在
理应
理当
理该
瑟瑟
甚且
甚么
甚或
甚而
甚至
甚至于
用
用来
甭
由
由于
由是
由此
由此可见
略为
略加
略微
的
的确
的话
皆可
目前
直到
直接
相似
相信
相反
相同
相对
相对而言
相应
相当
相等
省得
看
看上去
看出
看到
看来
看样子
看看
看见
看起来
真是
真正
眨眼
着
着呢
矣
矣乎
矣哉
知道
砰
确定
碰巧
离
种
积极
移动
究竟
穷年累月
突出
突然
立刻
立即
立地
立时
立马
竟然
竟而
第
第二
等
等到
等等
策略地
简直
简而言之
简言之
管
类如
精光
紧接着
累年
累次
纯粹
纵
纵令
纵使
纵然
练习
组成
经
经常
经过
结合
结果
给
绝不
绝对
绝非
绝顶
继之
继后
继续
继而
维持
综上所述
缕缕
罢了
老是
老老实实
考虑
者
而
而且
而况
而又
而后
而外
而已
而是
而言
而论
联系
联袂
背地里
背靠背
能
能否
能够
腾
自
自个儿
自从
自各儿
自后
自家
自己
自打
自身
至
至于
至今
至若
致
般的
良好
若
若夫
若是
若果
若非
范围
莫不
莫不然
莫如
莫若
莫非
获得
藉以
虽
虽则
虽然
虽说
行为
行动
表明
表示
被
要
要不
要不是
要不然
要么
要是
要求
规定
觉得
譬喻
譬如
认为
认真
认识
让
许多
论
论说
设使
设或
设若
诚如
诚然
话说
该
该当
说明
说来
说说
请勿
诸
诸位
诸如
谁
谁人
谁料
谁知
豁然
贼死
赖以
赶
赶快
赶早不赶晚
起
起先
起初
起头
起来
起见
起首
趁
趁便
趁势
趁早
趁机
趁热
趁着
越是
距
跟
路经
转动
转变
转贴
轰然
较
较为
较之
较比
边
达到
达旦
迅速
过
过于
过去
过来
运用
近几年来
近年来
近来
还
还是
还有
还要
这
这一来
这个
这么
这么些
这么样
这么点儿
这些
这会儿
这儿
这就是说
这时
这样
这次
这点
这种
这般
这边
这里
这麽
进入
进去
进来
进步
进而
进行
连
连同
连声
连日
连日来
连袂
连连
迟早
迫于
适应
适当
适用
逐步
逐渐
通常
通过
造成
遇到
遭到
遵循
遵照
避免
那
那个
那么
那么些
那么样
那些
那会儿
那儿
那时
那末
那样
那般
那边
那里
那麽
部分
都
鄙人
采取
里面
重大
重新
重要
鉴于
针对
长期以来
长此下去
长话短说
问题
间或
防止
阿
附近
限制
陡然
除
除了
除却
除去
除外
除开
除此
除此之外
除此以外
除此而外
除非
随
随后
随时
随着
随著
隔夜
隔日
难得
难怪
难说
难道
难道说
集中
需要
非但
非常
非徒
非得
非特
非独
靠
顶多
顷刻
顷刻之间
顷刻间
顺
顺着
顿时
风雨无阻
首先
马上
高低
高兴
默然
默默地
！
＃
％
＆
＇
（
）
）÷（１－
）、
＊
＋
＋ξ
＋＋
，
，也
－
－β
－－
－［＊］－
．
／
０：２
１．
１２％
２．３％
５：０
：
；
＜
＜±
＜Δ
＜λ
＜φ
＜＜
＝
＝″
＝☆
＝（
＝－
＝［
＝｛
＞
＞λ
？
Ａ
ＬＩ
Ｒ．Ｌ．
ＺＸＦＩＴＬ
［
［①①］
［①②］
［①③］
［①④］
［①⑤］
［①⑥］
［①⑦］
［①⑧］
［①⑨］
［①Ａ］
［①Ｂ］
［①Ｃ］
［①Ｄ］
［①Ｅ］
［①］
［①ａ］
［①ｃ］
［①ｄ］
［①ｅ］
［①ｆ］
［①ｇ］
［①ｈ］
［①ｉ］
［①ｏ］
［②
［②①］
［②②］
［②③］
［②④
［②⑤］
［②⑥］
［②⑦］
［②⑧］
［②⑩］
［②Ｂ］
［②Ｇ］
［②］
［②ａ］
［②ｂ］
［②ｃ］
［②ｄ］
［②ｅ］
［②ｆ］
［②ｇ］
［②ｈ］
［②ｉ］
［②ｊ］
［③①］
［③⑩］
［③Ｆ］
［③］
［③ａ］
［③ｂ］
［③ｃ］
［③ｄ］
［③ｅ］
［③ｇ］
［③ｈ］
［④］
［④ａ］
［④ｂ］
［④ｃ］
［④ｄ］
［④ｅ］
［⑤］
［⑤］］
［⑤ａ］
［⑤ｂ］
［⑤ｄ］
［⑤ｅ］
［⑤ｆ］
［⑥］
［⑦］
［⑧］
［⑨］
［⑩］
［＊］
［－
［］
］
］∧′＝［
］［
＿
ａ］
ｂ］
ｃ］
ｅ］
ｆ］
ｎｇ昉
｛－
｝
｝＞
～
～±
～＋
"""

    /// 内置停用词 = 精选中文功能词 ∪ 汇总的中英文停用词表。
    /// 刻意保留「研究 / 发展 / 社会 / 中国」等可能是主题词的实词，避免误伤专业词汇。
    static let builtinStopwords: Set<String> = {
        var s = legacyBuiltinStopwords
        s.formUnion(Set(stopwordsFullRaw.split(whereSeparator: \.isNewline).map(String.init)))
        return s
    }()

    /// 停用词表内容指纹：词表任何变化都会让依赖特征的缓存自动失效。
    static let stopwordsVersion: String = {
        var h: UInt64 = 0xCBF29CE484222325
        for w in builtinStopwords.sorted() {
            for b in w.utf8 { h = (h ^ UInt64(b)) &* 0x100000001B3 }
            h = (h ^ 0xFF) &* 0x100000001B3
        }
        return String(h, radix: 16)
    }()

    /// 出现在词首/词尾时，几乎必然是跨词边界的碎片（「的意」「是一种」「为主」）。
    static let weakBoundaryChars: Set<Character> = Set("的 了 着 过 得 地 是 在 和 与 或 也 就 都 而 之 以 于 其 这 那 一 不 很 更 最 可 能 会 要 将 被 把 从 对 为 所 由 并 且 但 则 者 矣 焉 乎 哉 又 再 还 才 只 已 曾 就 也 很 更 最 其 之 而 且 但 则 乃 若 即 便 却 竟 偏".filter { !$0.isWhitespace })

    static func loadExtra(_ path: String) -> Set<String> {
        guard !path.isEmpty,
              let text = try? String(contentsOfFile: (path as NSString).expandingTildeInPath,
                                     encoding: .utf8) else { return [] }
        return Set(text.split(separator: "\n").map(String.init).map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty && !$0.hasPrefix("#") })
    }

    /// 词首/词尾是虚字 → 跨词片段。仅对 ≥2 字词生效，避免误伤单字术语。
    static func hasWeakBoundary(_ word: String) -> Bool {
        guard let first = word.first, let last = word.last else { return false }
        return weakBoundaryChars.contains(first) || weakBoundaryChars.contains(last)
    }
}

// MARK: - 长词优先（子串碎片抑制）

/// 单个候选词参与子串判定的最小指标集。抽成纯函数便于单测构造样本。
struct WordMetric {
    let word: String
    let freq: Int
    let npmi: Double
    let hl: Double
    let hr: Double
}

enum SubstringFilter {
    /// 返回应被剔除的下标集合。以下信号全部满足才判为碎片：
    ///   (a) 长度：短词 ≥ 3 字、长词 ≥ 4 字。2 字是中文基本词汇单位，天然
    ///       常作复合词前后缀（商业银行 / 小额贷款），不能因「可扩展」判碎片；
    ///   (b) 覆盖：长词频次 ≥ 短词频次 × ratio；
    ///   (c) 凝固度：短词 NPMI < npmiCeiling 且低于长词
    ///       —— 区分「第三方/股权/余额」(≈0.8，真词) 与
    ///          「业的发展/速增长/款利率」(≈0.3~0.5，切错边界)；
    ///   (d) 边界：短词被粘住那侧熵 < gluedCeiling（邻居几乎固定），
    ///       且 < 长词同侧 × gluedRatio（相对也更受限）。
    static func indicesToRemove(_ items: [WordMetric], ratio: Double,
                                npmiCeiling: Double = 0.6,
                                gluedCeiling: Double = 1.8,
                                gluedRatio: Double = 0.8) -> Set<Int> {
        var byWord: [String: Int] = [:]
        for (pos, item) in items.enumerated() { byWord[item.word] = pos }
        var removed = Set<Int>()
        for item in items {
            let chars = Array(item.word)
            guard chars.count >= 4 else { continue }
            for len in 3..<chars.count {
                for start in 0...(chars.count - len) {
                    let sub = String(chars[start..<(start + len)])
                    guard let pos = byWord[sub], sub != item.word else { continue }
                    let short = items[pos]
                    guard Double(item.freq) >= Double(short.freq) * ratio else { continue }
                    guard short.npmi < npmiCeiling, short.npmi < item.npmi else { continue }

                    let isPrefix = start == 0
                    let isSuffix = start + len == chars.count
                    // 后缀 → 左邻被长词前一字固定；前缀 → 右邻被后续字固定；
                    // 中间 → 两侧都被固定。
                    let glued: Double
                    let longSameSide: Double
                    if isPrefix && !isSuffix {
                        glued = short.hr
                        longSameSide = item.hr
                    } else if isSuffix && !isPrefix {
                        glued = short.hl
                        longSameSide = item.hl
                    } else {
                        glued = min(short.hl, short.hr)
                        longSameSide = min(item.hl, item.hr)
                    }
                    if glued < gluedCeiling && glued < longSameSide * gluedRatio {
                        removed.insert(pos)
                    }
                }
            }
        }
        return removed
    }
}

// MARK: - 提取器

enum ExtractError: Error, CustomStringConvertible {
    case invalidRange(String)
    case missingInput(String)

    var description: String {
        switch self {
        case .invalidRange(let msg), .missingInput(let msg): return msg
        }
    }
}

struct Candidate {
    let key: GramKey
    let count: Int
    let pmi: Double
    let npmi: Double
}

enum CoreWordExtractor {

    static func isHan(_ ch: Character) -> Bool {
        guard ch.unicodeScalars.count == 1, let v = ch.unicodeScalars.first?.value else {
            return false
        }
        return (v >= 0x4E00 && v <= 0x9FFF)
            || (v >= 0x3400 && v <= 0x4DBF)
            || (v >= 0x20000 && v <= 0x2A6DF)
            || (v >= 0xF900 && v <= 0xFAFF)
            || (v >= 0x2F800 && v <= 0x2FA1F)
    }

    static func isASCIINumOrLetter(_ ch: Character) -> Bool {
        guard ch.unicodeScalars.count == 1, let v = ch.unicodeScalars.first?.value else {
            return false
        }
        return (v >= 0x30 && v <= 0x39)
            || (v >= 0x41 && v <= 0x5A)
            || (v >= 0x61 && v <= 0x7A)
    }

    /// 便于自检/外部阅读：把正文切成连续汉字字符串片段。
    static func segments(_ text: String, keepASCII: Bool = false) -> [String] {
        var result: [String] = []
        var buf: [Character] = []
        for ch in text {
            if isHan(ch) || (keepASCII && isASCIINumOrLetter(ch)) {
                buf.append(ch)
            } else if !buf.isEmpty {
                result.append(String(buf))
                buf.removeAll(keepingCapacity: true)
            }
        }
        if !buf.isEmpty { result.append(String(buf)) }
        return result
    }

    /// 把全文编码成单条 [UInt16] 流：真实字符 ID ≥ 1，0 表示片段边界。
    static func encode(_ text: String, keepASCII: Bool, maxChars: Int,
                       codec: inout GramCodec) -> (stream: [UInt16], totalChars: Int, segments: Int) {
        var stream: [UInt16] = []
        let cap = min(text.count, maxChars > 0 ? maxChars + 1 : 20_000_000)
        stream.reserveCapacity(cap)
        var total = 0
        var segCount = 0
        var inSegment = false
        for ch in text {
            if isHan(ch) || (keepASCII && isASCIINumOrLetter(ch)) {
                if maxChars > 0 && total >= maxChars { break }
                stream.append(codec.id(for: ch))
                total += 1
                if !inSegment { segCount += 1; inSegment = true }
            } else if inSegment {
                stream.append(sentinelID)
                inSegment = false
            }
        }
        if inSegment { stream.append(sentinelID) }
        return (stream, total, segCount)
    }

    /// 每个位置起连续非 0 字符数（上限 maxN），用于避免跨片段构词。
    static func runLengths(_ stream: [UInt16], maxN: Int) -> [UInt8] {
        var runs = [UInt8](repeating: 0, count: stream.count)
        var run = 0
        for i in stride(from: stream.count - 1, through: 0, by: -1) {
            if stream[i] == sentinelID {
                run = 0
            } else {
                run = min(maxN, run + 1)
            }
            runs[i] = UInt8(run)
        }
        return runs
    }

    /// 频次表：UInt32 计数（单本书不会超 40 亿次），比 Int 字典省一半 value 空间。
    static func countNgrams(_ stream: [UInt16], runs: [UInt8], maxN: Int)
        -> [GramKey: UInt32] {
        var freq: [GramKey: UInt32] = [:]
        for i in 0..<stream.count {
            let nMax = Int(runs[i])
            if nMax == 0 { continue }
            for n in 1...nMax {
                let key = GramKey.make(stream, i, n)
                freq[key, default: 0] += 1
            }
        }
        return freq
    }

    /// NPMI：所有二分切分点取最小 PMI，再除以 -log p(w)。
    static func wordNPMI(_ key: GramKey, freq: [GramKey: UInt32], total: Int)
        -> (pmi: Double, npmi: Double) {
        guard total > 0 else { return (0.0, -Double.infinity) }
        let ids = key.ids()
        let n = ids.count
        guard n >= 2, let c32 = freq[key] else { return (0.0, -Double.infinity) }
        let c = Double(c32)
        let pw = c / Double(total)
        guard pw > 0, pw < 1 else { return (0.0, -Double.infinity) }

        var minPMI = Double.infinity
        for k in 1..<n {
            let left = GramKey.make(Array(ids[0..<k]))
            let right = GramKey.make(Array(ids[k..<n]))
            guard let lp = freq[left], let rp = freq[right] else {
                return (0.0, -Double.infinity)
            }
            let pl = Double(lp) / Double(total)
            let pr = Double(rp) / Double(total)
            guard pl > 0, pr > 0 else { return (0.0, -Double.infinity) }
            minPMI = min(minPMI, log(pw / (pl * pr)))
        }
        guard minPMI.isFinite else { return (0.0, -Double.infinity) }
        return (minPMI, minPMI / (-log(pw)))
    }

    /// H = log2 T - Σ(c·log2 c)/T，只需两个累加量，无需保存邻居分布。
    @inline(__always)
    static func entropy(total: Int, sumCLogC: Double) -> Double {
        guard total > 0 else { return 0.0 }
        let t = Double(total)
        return max(0.0, log2(t) - sumCLogC / t)
    }

    /// 第二遍：扁平记录 (candID, 左/右邻字 ID) → 次数。
    static func countNeighbors(
        _ stream: [UInt16], runs: [UInt8], candidateIndex: [GramKey: Int32],
        ngramMin: Int, ngramMax: Int, boundary: ExtractOptions.BoundaryMode
    ) -> (left: [UInt64: Int32], right: [UInt64: Int32]) {
        var left: [UInt64: Int32] = [:]
        var right: [UInt64: Int32] = [:]
        let size = stream.count
        for i in 0..<size {
            let nMax = min(ngramMax, Int(runs[i]))
            if nMax < ngramMin { continue }
            for n in ngramMin...nMax {
                let key = GramKey.make(stream, i, n)
                guard let cidValue = candidateIndex[key] else { continue }
                let cid = UInt64(cidValue) << 16

                if i > 0 {
                    let ch = stream[i - 1]
                    if ch != sentinelID || boundary == .count {
                        let k = cid | UInt64(ch)
                        left[k, default: 0] += 1
                    }
                } else if boundary == .count {
                    left[cid, default: 0] += 1
                }

                if i + n < size {
                    let ch = stream[i + n]
                    if ch != sentinelID || boundary == .count {
                        let k = cid | UInt64(ch)
                        right[k, default: 0] += 1
                    }
                } else if boundary == .count {
                    right[cid, default: 0] += 1
                }
            }
        }
        return (left, right)
    }

    static func extract(text: String, options: ExtractOptions) throws -> ExtractionResult {
        guard options.ngramMin >= 2, options.ngramMax >= options.ngramMin,
              options.ngramMax <= 5 else {
            throw ExtractError.invalidRange("需满足 2 <= --ngram-min <= --ngram-max <= 5")
        }

        var codec = GramCodec()
        let encoded = encode(text, keepASCII: options.keepASCII,
                             maxChars: options.maxChars, codec: &codec)
        let stream = encoded.stream
        let total = encoded.totalChars
        let runs = runLengths(stream, maxN: options.ngramMax)
        let freq = countNgrams(stream, runs: runs, maxN: options.ngramMax)

        // 等价剪枝：任一熵阈值 > 0 时，频次 1 的 gram 该侧熵必为 0 → 必被淘汰。
        let entropyActive = options.minLeftEntropy > 0 || options.minRightEntropy > 0
        let effectiveMinFreq = max(options.minFreq, entropyActive ? 2 : 1)

        var candidates: [Candidate] = []
        var candidateIndex: [GramKey: Int32] = [:]
        var ngramTypes = 0
        for (key, c32) in freq {
            let length = key.length
            guard length >= options.ngramMin, length <= options.ngramMax else { continue }
            ngramTypes += 1
            let c = Int(c32)
            guard c >= effectiveMinFreq else { continue }
            let (pmi, npmi) = wordNPMI(key, freq: freq, total: total)
            guard npmi >= options.minNPMI else { continue }
            let cid = candidates.count
            candidates.append(Candidate(key: key, count: c, pmi: pmi, npmi: npmi))
            candidateIndex[key] = Int32(cid)
        }

        let (leftFlat, rightFlat) = countNeighbors(
            stream, runs: runs, candidateIndex: candidateIndex, ngramMin: options.ngramMin,
            ngramMax: options.ngramMax, boundary: options.boundary)

        // 把扁平计数聚合到每个候选的 Σc·log2c 与 T。
        var leftTotal = [Int](repeating: 0, count: candidates.count)
        var rightTotal = [Int](repeating: 0, count: candidates.count)
        var leftSum = [Double](repeating: 0, count: candidates.count)
        var rightSum = [Double](repeating: 0, count: candidates.count)
        for (key, c) in leftFlat {
            let cid = Int(key >> 16)
            guard cid >= 0, cid < candidates.count else { continue }
            leftTotal[cid] += Int(c)
            leftSum[cid] += Double(c) * log2(Double(c))
        }
        for (key, c) in rightFlat {
            let cid = Int(key >> 16)
            guard cid >= 0, cid < candidates.count else { continue }
            rightTotal[cid] += Int(c)
            rightSum[cid] += Double(c) * log2(Double(c))
        }

        struct Survived {
            let index: Int
            let word: String
            let freq: Int
            let pmi: Double
            let npmi: Double
            let hl: Double
            let hr: Double
        }
        var survived: [Survived] = []
        for (idx, cand) in candidates.enumerated() {
            let hl = entropy(total: leftTotal[idx], sumCLogC: leftSum[idx])
            let hr = entropy(total: rightTotal[idx], sumCLogC: rightSum[idx])
            guard hl >= options.minLeftEntropy, hr >= options.minRightEntropy else { continue }
            survived.append(Survived(index: idx, word: codec.word(cand.key), freq: cand.count,
                                     pmi: cand.pmi, npmi: cand.npmi, hl: hl, hr: hr))
        }

        // 停用词 + 首尾虚字碎片。
        var stopwords = options.useStopwords ? WordFilters.builtinStopwords : []
        if options.useStopwords { stopwords.formUnion(WordFilters.loadExtra(options.stopwordsPath)) }
        var stopwordRemoved = 0
        var boundaryRemoved = 0
        var kept: [Survived] = []
        for item in survived {
            if options.useStopwords && stopwords.contains(item.word) {
                stopwordRemoved += 1
                continue
            }
            if options.useBoundaryFilter && WordFilters.hasWeakBoundary(item.word) {
                boundaryRemoved += 1
                continue
            }
            kept.append(item)
        }

        // 长词优先：抑制被更长词覆盖、内部松散、边界被粘住的子串碎片。
        var substringRemoved = 0
        if options.useSubstringFilter {
            let metrics = kept.map {
                WordMetric(word: $0.word, freq: $0.freq, npmi: $0.npmi, hl: $0.hl, hr: $0.hr)
            }
            let removed = SubstringFilter.indicesToRemove(metrics, ratio: options.substringRatio)
            substringRemoved = removed.count
            kept = kept.enumerated().filter { !removed.contains($0.offset) }.map(\.element)
        }

        let bonus = options.lengthBonus
        let words: [CoreWord] = kept
            .map { item in
                (item: item,
                 score: Double(item.freq) * (1 + bonus * Double(item.word.count - 2)))
            }
            .sorted { a, b in
                if a.score != b.score { return a.score > b.score }
                if a.item.word.count != b.item.word.count {
                    return a.item.word.count > b.item.word.count
                }
                if a.item.npmi != b.item.npmi { return a.item.npmi > b.item.npmi }
                return a.item.word < b.item.word
            }
            .prefix(max(0, options.top))
            .map { CoreWord(word: $0.item.word, freq: $0.item.freq, pmi: $0.item.pmi,
                            npmi: $0.item.npmi, leftEntropy: $0.item.hl,
                            rightEntropy: $0.item.hr) }

        let stats = ExtractionResult.Stats(
            totalChars: total,
            totalSegments: encoded.segments,
            ngramTypes: ngramTypes,
            npmiSurvivors: candidates.count,
            entropySurvivors: survived.count,
            stopwordRemoved: stopwordRemoved,
            boundaryRemoved: boundaryRemoved,
            substringRemoved: substringRemoved,
            returned: words.count)
        return ExtractionResult(words: words, stats: stats, params: options)
    }
}

// MARK: - 输出

enum OutputFormat: String {
    case table, words, json
}

enum ResultFormatter {
    static func render(_ result: ExtractionResult, format: OutputFormat) throws -> String {
        switch format {
        case .words:
            return result.words.map(\.word).joined(separator: "\n")
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            let data = try encoder.encode(result)
            return String(data: data, encoding: .utf8) ?? "{}"
        case .table:
            var lines = ["rank\tword\tfreq\tnpmi\tpmi\tH_left(bit)\tH_right(bit)"]
            for (rank, c) in result.words.enumerated() {
                lines.append(String(
                    format: "%d\t%@\t%d\t%.4f\t%.4f\t%.4f\t%.4f",
                    rank + 1, c.word, c.freq, c.npmi, c.pmi, c.leftEntropy, c.rightEntropy))
            }
            return lines.joined(separator: "\n")
        }
    }

    static func statsLine(_ result: ExtractionResult) -> String {
        let s = result.stats
        return "[core-words] 字数=\(s.totalChars) 片段=\(s.totalSegments) "
            + "N-gram种类=\(s.ngramTypes) NPMI通过=\(s.npmiSurvivors) "
            + "熵通过=\(s.entropySurvivors) 停用词剔除=\(s.stopwordRemoved) "
            + "虚字碎片=\(s.boundaryRemoved) 子串碎片=\(s.substringRemoved) "
            + "输出=\(s.returned)"
    }
}

// MARK: - 语料加载

enum CorpusLoader {
    static func read(_ path: URL, encoding: String) throws -> String {
        let data = try Data(contentsOf: path)
        if encoding != "auto", let enc = encodingByName(encoding) {
            return String(data: data, encoding: enc)
                ?? String(decoding: data, as: UTF8.self)
        }
        for enc in [String.Encoding.utf8, gb18030, String.Encoding.isoLatin1] {
            if var text = String(data: data, encoding: enc) {
                if text.hasPrefix("\u{FEFF}") { text.removeFirst() }
                return text
            }
        }
        return String(decoding: data, as: UTF8.self)
    }

    static func encodingByName(_ name: String) -> String.Encoding? {
        switch name.lowercased() {
        case "utf-8", "utf8": return .utf8
        case "gb18030", "gbk", "gb2312": return gb18030
        case "latin1", "iso-8859-1": return .isoLatin1
        default: return nil
        }
    }

    static var gb18030: String.Encoding {
        #if canImport(CoreFoundation)
        let cf = CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)
        return String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cf))
        #else
        return .utf8
        #endif
    }

    static func load(_ source: String, encoding: String) throws -> String {
        if source == "-" {
            let data = FileHandle.standardInput.readDataToEndOfFile()
            for enc in [String.Encoding.utf8, gb18030, .isoLatin1] {
                if let text = String(data: data, encoding: enc) { return text }
            }
            return String(decoding: data, as: UTF8.self)
        }
        let path = URL(fileURLWithPath: (source as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir) else {
            throw ExtractError.missingInput("输入不存在: \(source)")
        }
        if !isDir.boolValue { return try read(path, encoding: encoding) }

        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: path, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return "" }
        var files: [URL] = []
        for case let url as URL in enumerator where textSuffixes.contains(url.pathExtension.lowercased()) {
            files.append(url)
        }
        files.sort { $0.path < $1.path }
        return try files.map { try read($0, encoding: encoding) }.joined(separator: "\n")
    }
}

