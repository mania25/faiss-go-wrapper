package main

import "C"

func main() {
	faissObject := New()
	faissObject.InitFaissDB()

	vectors := []C.float{0.04781779779919556, -0.002729872108570167, 0.24663587020976202, 0.2044068469632683, -0.17525611205824784, -0.05198291742375919, -0.12790788177933013, 0.08509331768644708, -0.0012752711772918701, 0.04391562991908619, -0.17724104119198664, -0.14059846475720406, 0.447689903434366, 0.21604224400860922, -0.24540016361113107, 0.20767887362412044, 0.011388738772698812, 0.03280311556799071, 0.12463166724358286, 0.2992921258722033, -0.10743052991373199, -0.368512715612139, -0.23048570682294667, -0.5299933552742004, -0.06635599989177925, -0.00961737853607961, 0.3963915040450437, -0.1013873927295208, 0.1404768473335675, 0.2300938612648419, -0.004532435908913612, -0.13150714178170478, -0.08004367378141199, -0.3524852198149477, -0.3572475388646126, 0.17056549074394362, -0.011814217482294356, 0.25619328607405933, -0.06545104937894004, 0.144539225846529, -0.013257602761898721, -0.45706415602139067, -0.10290951920407158, 0.08854409839425768, -0.4424314775637218, 0.020379215279327973, 0.0031899724687848774, -0.00666448420711926, 0.06468055796410356, 0.32259821040289743, 0.07165127832974706, 0.21235173381865025, 0.468235687485763, -0.018652809517724172, 0.10383635360215392, 0.018278990206973895, -0.026322584732302597, 0.1459354603929179, -0.22694522142410278, -0.2734861373901367, -0.019513959331171855, 0.15919623098203114, -0.20163607597351074, -0.3053913738445512, -0.1836792959698609, 0.14598490750151022, -0.022185774786131724, -0.26175486110150814, -0.5123070040717721, -0.6878506158079419, -0.042068747005292347, 0.1428519913128444, 0.003516478730099542, 0.20983438034142768, -0.1061650099498885, -0.1600796122636114, 0.24621375064764703, -0.14518324977585248, 0.35717511576201233, -0.3931745375905718, 0.07653065132243293, 0.029263562389782498, 0.14613263309001923, 0.09379332193306514, -0.2425340073449271, -0.028666310611047914, 0.18401740491390228, 0.09611746282981974, 0.36024320338453564, -0.3522510251828602, -0.006748615631035396, -0.4170674889215401, -0.12328253633209638, 0.17661243570702417, 0.2971255260386637, 0.1656305811234883, -0.32282878192407743, 0.2326322772673198, 0.005645498633384705, -0.044537269111190526, -0.0660029234630721, 0.033258259030325074, 0.29602675459214617, -0.1385599821805954, -0.4252705270690577, -0.3591596218092101, 0.10074759354548794, -0.02764349856546947, -0.21082833941493714, 0.21966033848002553, -0.19659017718264035, 0.37286974037332193, 0.30920843141419546, -0.13892923880900657, -0.23101425901404582, -0.1291863269039563, -0.3022397815116814, 0.04991209653339216, 0.35490932528461727, -0.04323679953813553, -0.12101671312536512, -0.1775551438331604, 0.20274909672194294, 0.004757295365769616, -0.2706222805593695, 0.17736943598304475, 0.320711004946913, 0.16157939870442664, 0.0715829870234171, -0.029289402676924706, 0.12208139313897955, 0.15110377053532215, -0.1664823454464535, 0.02338948148765607, 0.047897594155706075, 0.0966707800768914, 0.07012689817825075, 0.21353480247027815, -0.002895786477152459, -0.09780270432980877, 0.3150465617954597, 0.15149296830201933, -0.31356072305941474, 0.19843873741317117, 0.14108044476893836, 0.13518239633029627, 0.1442247328507396, 0.22740839389682618, -0.08937466270597039, -0.18776099411575514, 0.026679272673505464, -0.3113952361292646, -0.08946041635957039, 0.04624912236025278, 0.20034537375779143, -0.12596930925383004, 0.14356434091606352, 0.07076099485961977, -0.12387461156283176, -0.11061475772416209, 0.011818220412280566, -0.2509234958331095, -0.11429797309244465, 0.0407330696453608, 0.23771904904149962, 0.11224486588982832, -0.20615531500808137, 0.18515196258361133, 0.18028896833033842, -0.37803603059885477, -0.20957738158856323, 0.11565740819787607, -0.2959587953282063, -0.1301367391057, 0.14382938709121176, -0.12142229902697099, 0.05081714912558889, 0.12822850990077078, 0.18069057096587773, 0.1494267601268868, 0.15935996025075724, 0.0022552028761250887, 0.2341972193520371, 0.12235163807951564, -0.06808317014334088, 0.1375092627009388, -0.19593279092143218, -0.17377648751312744, -0.02170826058519703, 0.006684813958670788, 0.01909900582981056, -0.35194324592719534, -0.2311109038276185, -0.052694619979017865, -0.12943807074696964, -0.20920599342329416, -0.20181644484010375, -0.4005479834036913, 0.037078365479305965, 0.1926535177741668, 0.002645258780939694, 0.2598993075944617, -0.1096771085010228, -0.04947413090675021, 0.2365593132545508, -0.07737131602225636, 0.09757771587290684, -0.15706011411192977, 0.04680465267201562, 0.19735517479089984, 0.03682053318974969, 0.07609937932972467, -0.04686722657749566, -0.0013513622350968466, 0.13287940693316524, 0.011686794449363328, 0.29125392694628527, -0.1492022622410141, 0.10274937432705464, -0.11885190147259635, 0.0660720939704769, 0.21179723830804406, 0.058818028711025115, 0.12230804241485448, 0.006189208609138443, 0.1764496033997303, 0.04934513170695947, 0.0611656491427186, -0.01979838552868027, 0.13635778687838576, 0.23426875618354748, -0.3117605442722177, -0.3461783065873304, -0.11756036396021258, 0.10309854976506624, -0.17297794566696098, -0.10685505469860844, 0.18844479748778448, -0.14890881432331876, 0.14356739051684647, 0.11508350538035497, -0.04319265304261562, 0.025556761336277243, -0.18310970388015477, -0.12758561205899643, 0.03035957524051031, 0.1619400099480335, -0.20726364213292917, -0.14584213756247907, -0.08852952491759719, 0.02010037403408728, 0.12281160374532954, -0.14863280213610855, 0.03399748566231417, 0.13492131573677865, 0.08562972459211544}
	faissObject.PushTrainDataVector(vectors)

	vectors = []C.float{-0.32089716879030067, -0.1069410862789179, 0.729600672920545, 0.6884424462914467, 0.41438760546346504, 0.17045519749323526, -0.263383474200964, -0.038151226937770844, -0.029358378301064175, 0.4733591716115673, 0.5888127833604813, 0.16712415218353271, -0.04320811294019222, 0.34772024552027386, -0.6035002296169599, 0.3721342794597149, 0.2023953308040897, 0.2605406989653905, -0.1927919089794159, 0.4101245030760765, 0.07816521691468854, 0.4030055928354462, 0.30012652184814215, -0.914093396315972, -0.3369816889365514, -0.16658224165439606, 0.13348488012949625, -0.40125425656636554, 0.1752246500303348, 0.3997866188486417, -0.021188676357269287, -0.029557447880506516, -0.06628502625972033, -0.4093582158287366, 0.09577826860671242, 0.3173888909320037, -0.39089278876781464, 0.24374636945625147, -0.4391282933453719, 0.07112231416006883, -0.018612128992875416, -0.5034487197796503, -0.3076343710223834, -0.06258751824498177, -0.09262352933486302, -0.11309502522150676, 0.22415871421496072, 0.4418135980765025, -0.20426652378713092, -0.24733485778172812, 0.23667432072882852, -0.05330572774012884, 0.1945488266646862, 0.02714777986208598, 0.4006592432657878, 0.30008570353190106, 0.26484792431195575, 0.26922692358493805, -0.45731352269649506, -0.46607182795802754, -0.07517476255695026, -0.2726062110935648, 0.43097875267267227, -0.6100466474890709, -0.5955844769875208, 0.09000048196564119, -0.10564573450634877, 0.17230804761250815, -0.47976432492335636, -0.492655170460542, 0.1507826562349995, 0.2542043849825859, 0.188779661936375, -0.08915642268645267, 0.08254272490739822, 0.11217912038167317, -0.01789526641368866, 0.24098037870135158, 0.39919992287953693, -0.35305979785819847, -0.3583262798686822, 0.3569515962153673, -0.2067698184400797, 0.014928478747606277, 0.09753426461247727, 0.06744650689264138, -0.4017684596280257, -0.4383562095463276, 0.39333827296892804, 0.1827098379532496, 0.4294539491335551, 0.156843694858253, -0.01696951314806938, 0.4929940759514769, 0.23732798384298803, -0.30639927337567013, -0.17700336873531342, -0.28659413506587345, -0.08597026749824484, 0.2039186026280125, 0.12373115681111813, 0.4321497728427251, 0.4060914081831773, 0.012366446355978647, -0.1657134878138701, -0.21132741371790567, -0.07914551378538211, -0.3699440819521745, 0.3291587394972642, 0.18745089570681253, -0.03576091273377339, -3.6170706152915955e-05, -0.14250576961785555, 0.03784205640355746, 0.2177735318740209, -0.005278981601198514, -0.7084795931975046, -0.17051173249880472, -0.017562629655003548, 0.09016463160514832, -0.21255430082480112, -0.3590414136027296, -0.45794477810462314, 0.19629231871416172, 0.17400705193479857, -0.07680519173542659, 0.042293267945448555, -0.0023512461533149085, -0.04250923288054764, -0.10071689784268124, 0.26976013579405844, 0.5183708748987151, -0.2727276261818285, 0.16424232979301856, -0.04963801410566601, 0.1842005060429478, -0.0793843746101225, 0.21591317581219804, 0.2646704845311534, -0.09313208911852497, 0.006418689425724248, 0.015365141948374609, -0.30763616123133236, 0.1369679358645549, 0.17544078077965727, 0.17471055090168697, 0.06104580111180743, 0.3552327394265578, -0.11653960262063062, 0.12005313181240733, 0.17870981872248295, -0.2954410713993841, -0.19389867772244745, -0.012311580409813259, 0.2796598423091281, -0.25618080083060907, 0.3618100233660597, 0.007053085124223596, -0.15880720014683902, 0.0646571720070723, -0.03723767672717157, -0.38265736556301516, -0.18864797861574012, 0.16235662611304885, 0.2180724934183268, 0.25523304988423157, -0.2591295903645611, -0.04309904790069494, 0.36804381511562195, -0.4108903601558672, -0.10461823230919738, -0.05656593366034536, -0.2007308617596411, -0.07917145409414338, 0.22557751924937797, 0.06747562407205503, 0.03052808207252787, 0.08608475442613578, 0.23592998364215922, 0.13069461379846972, 0.10574795903327565, 0.0940387814771384, 0.24625651358575043, 0.4779488337226212, -0.10126771575758337, 0.14537584870736786, -0.3317737207851476, -0.12595394849389172, -0.16549942085597044, -0.23472705527415705, 0.11363822074296574, -0.5251772223661343, -0.4029193163668323, 0.1786274858103651, -0.24118351717091477, 0.0008327462451739444, -0.027238040425193805, -0.030144308228045702, -0.038015721876743354, 0.1726821158339994, 0.16602934362082225, 0.06985086327444555, -0.15346535895757066, 0.0799211829693781, 0.11172376111305009, -0.045883427409636274, 0.15018984041105593, -0.07524838647158402, 0.12526204264981466, 0.2580113917744408, -0.024415887167884245, -0.05826342262298567, 0.04817674504172626, 0.21386355031023008, 0.061030060708516326, -0.17988781543034646, 0.2679621244315058, -0.05778398488958677, 0.15194383637087108, -0.13898643546013367, 0.0012200898345327005, 0.3793295688616733, 0.20072106427910008, 0.04328046994891742, 0.1697901366601905, 0.07251055344628791, 0.1403812048698051, 0.04737340038991533, -0.02944180409475747, 0.22232577541015214, 0.2800196800380945, -0.2418604963976476, -0.141246354388487, 0.05825786178724633, 0.03237032295954931, -0.4363636304106977, 0.03214850367819761, 0.2754096147190366, -0.1704767357506272, 0.16014436843882626, -0.08050222161061053, -0.03804123396144456, 0.044191111224386584, -0.22245243675489393, -0.261038917905858, -0.1514007859966821, 0.11320651583890948, -0.12079589022646865, -0.3216849047069748, -0.23644639823275307, -0.21968519591933322, 0.2343305616846515, -0.04486836270532674, -0.0662560772928676, 0.1467444971203804, 0.03273485598361327};
	faissObject.PushTrainDataVector(vectors)

	faissObject.BuildIndex()
}